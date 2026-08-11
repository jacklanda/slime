"""Build a per-session training trajectory from multi-turn conversation data.

The :class:`TrajectoryManager` builds one trajectory per session. ``record_turn``
feeds in each turn (prompt messages + the served model's sglang snapshot),
routing it into a per-sid message tree; ``get_trajectory`` then linearizes that
tree into a ``list[Sample]`` of loss-masked training rows. When adapters provide
strict TITO evidence, the manager concatenates per-turn context deltas and
generated outputs without re-tokenizing or replacing historical tokens.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
from collections.abc import Iterator
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ===========================================================================
# TurnRecord
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One sglang ``/generate`` snapshot: the contract between an adapter and the
    manager. Adapters build it from a turn's prompt/output token ids; ``record_turn``
    consumes it."""

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    weight_version: str | None = None
    require_rollout_logprobs: bool = False
    require_weight_version: bool = False
    loss_mask: list[int] | None = None
    policy_loss_mask: list[int] | None = None
    context_delta_ids: list[int] | None = None
    tito_boundary_before: bool = False
    tito_model_type: str | None = None
    tito_context_reason: str | None = None
    disable_thinking: bool | None = None
    prompt_context_start_idx: int | None = None
    rollout_top_p_token_ids: list[int] | None = None
    rollout_top_p_token_offsets: list[int] | None = None
    ill_formed: bool = False


# ===========================================================================
# MessageNode
# ===========================================================================


class MessageNode:
    """One node in a session's routing tree, carrying a single chat message
    (``None`` for the dummy root and for an assistant leaf we generated but
    whose ``response_message`` was empty).

    The two kinds are distinguished by whether ``turn`` is set, which reflects
    WHERE the message came from:

    * **generated** (``turn is not None``): an assistant message the model
      actually generated this turn, fed in via ``record_turn``. ``turn`` holds
      its :class:`TurnRecord` -- the prompt/output ids, logprobs and finish
      reason that ``get_trajectory`` linearizes into training tokens.
    * **routing-only** (``turn is None``): the message came from the prompt, not
      from generation, so it only exists to route. This is every
      system/user/tool node, AND any assistant we did NOT generate: a foreign
      assistant the client replayed in a later prompt, or a prior generated turn
      demoted by the rewrite-merge in ``_try_merge_assistant_rewrite``.
    """

    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.metadata = dict(metadata or {})
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None  # the generated TurnRecord, else None (routing-only)
        self.turn_index: int | None = None
        # Shared by sibling leaf paths; the first to reach it trains on it, the rest
        # re-emit it as loss_mask=0 context -- so each response is trained exactly once.
        self.response_trained: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: MessageNode) -> MessageNode:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[MessageNode]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[MessageNode] = []
        node: MessageNode | None = self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[MessageNode]:
        stack = [self]
        while stack:
            node = stack.pop()
            if not node.children:
                yield node
                continue
            stack.extend(reversed(node.children))


# ===========================================================================
# drift classification — how an incoming turn's prompt relates to held tokens
# ===========================================================================


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        chunk_end = min(matched + chunk, limit)
        if a[matched:chunk_end] == b[matched:chunk_end]:
            matched = chunk_end
        else:
            while matched < chunk_end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


class DriftKind(enum.Enum):
    CLEAN = "clean"  # prompt_ids exactly extends held tokens, or strict context_delta_ids is present
    FORK = "fork"  # close this builder, open a fresh one as a segment boundary


# ===========================================================================
# SampleBuilder — accumulates turns into one trainable Sample (fork closes it)
# ===========================================================================


class _SampleBuilder:
    """Accumulates a chain's turns into the token sequence of one ``Sample``.

    A chain of turns is appended one at a time via :meth:`append_turn`. Strict
    TITO adapters provide ``context_delta_ids``: the exact non-trainable prompt
    tokens added since the previous generated output. In that mode the builder
    never inspects or rewrites historical prompt tokens; it just concatenates
    the delta and the generated response evidence. Older adapters without
    ``context_delta_ids`` fall back to exact-prefix prompt extension.

    * **CLEAN** -- append the strict context delta, or append the prompt tail when
      prompt_ids exactly extends held tokens.
    * **FORK** -- any unproven drift or explicit TITO boundary. The caller closes
      the current builder and opens a fresh one.

    Each surviving builder yields one Sample.
    """

    def __init__(self, fork_threshold: int, *, boundary_reason: str | None = None) -> None:
        self._fork_threshold = fork_threshold
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.policy_loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.top_p_token_ids: list[int] | None = None
        self.top_p_token_offsets: list[int] | None = None
        self.last_response_start_idx: int | None = None
        self.leading_prompt_len: int = 0
        self.weight_versions: set[str] = set()
        self.boundary_reason = boundary_reason
        self.tito_context_reasons: list[str] = []
        self.tito_model_types: set[str] = set()
        self.turn_spans: list[dict[str, Any]] = []

    def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
        """Decide how this builder should absorb ``turn``'s prompt.

        Strict TITO turns carry their own context delta. They are appendable unless
        the adapter explicitly marks a boundary before the turn. Legacy turns are
        appendable only when their prompt exactly extends the held tokens; any
        drift forks instead of realigning, so old rollout logprobs are never
        attached to replay-mutated token ids.
        """
        if turn.tito_boundary_before:
            return DriftKind.FORK
        if turn.context_delta_ids is not None:
            return DriftKind.CLEAN

        realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
        drift = len(self.tokens) - realign_at

        if drift == 0:
            return DriftKind.CLEAN

        return DriftKind.FORK

    def append_turn(
        self,
        turn: TurnRecord,
        kind: DriftKind,
        *,
        turn_index: int,
        trained: bool = True,
    ) -> None:
        """Append one turn into this SampleBuilder."""
        assert kind is not DriftKind.FORK, "append_turn called on a builder that would fork"

        is_first_turn = self.last_response_start_idx is None
        if turn.weight_version is not None:
            self.weight_versions.add(str(turn.weight_version))
        if turn.tito_context_reason is not None:
            self.tito_context_reasons.append(turn.tito_context_reason)
        if turn.tito_model_type is not None:
            self.tito_model_types.add(turn.tito_model_type)

        # --- append this turn's prompt tail (loss_mask=0) ---
        if turn.context_delta_ids is not None:
            self._append_tokens(turn.context_delta_ids, loss_mask=0)
        else:  # CLEAN: held tokens are an exact prefix of prompt_ids; append the tail beyond them
            self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)

        # --- append this turn's generated response (loss_mask=1 unless re-emitted as context) ---
        self.last_response_start_idx = len(self.tokens)
        response_start = self.last_response_start_idx
        if trained:
            response_mask = turn.loss_mask if turn.loss_mask is not None else 1
            policy_response_mask = turn.policy_loss_mask if turn.policy_loss_mask is not None else response_mask
            response_logprobs = self._masked_logprobs(turn.output_log_probs, response_mask, len(turn.output_ids))
            self._append_tokens(
                turn.output_ids,
                loss_mask=response_mask,
                policy_loss_mask=policy_response_mask,
                logprobs=response_logprobs,
                top_p_token_ids=turn.rollout_top_p_token_ids,
                top_p_token_offsets=turn.rollout_top_p_token_offsets,
            )
        else:
            self._append_tokens(turn.output_ids, loss_mask=0)
        response_end = len(self.tokens)
        self.turn_spans.append(
            {
                "turn_index": turn_index,
                "response_token_start": response_start,
                "response_token_end": response_end,
                "trained": bool(trained),
            }
        )

        if is_first_turn:
            self.leading_prompt_len = self.last_response_start_idx

    def _append_tokens(
        self,
        ids: list[int],
        *,
        loss_mask: int | list[int],
        policy_loss_mask: int | list[int] | None = None,
        logprobs: list[float] | None = None,
        top_p_token_ids: list[int] | None = None,
        top_p_token_offsets: list[int] | None = None,
    ) -> None:
        self.tokens.extend(ids)
        if isinstance(loss_mask, list):
            assert len(loss_mask) == len(ids), f"loss_mask length {len(loss_mask)} != ids length {len(ids)}"
            self.loss_mask.extend(loss_mask)
        else:
            self.loss_mask.extend([loss_mask] * len(ids))
        if policy_loss_mask is None:
            policy_loss_mask = loss_mask
        if isinstance(policy_loss_mask, list):
            assert len(policy_loss_mask) == len(
                ids
            ), f"policy_loss_mask length {len(policy_loss_mask)} != ids length {len(ids)}"
            self.policy_loss_mask.extend(policy_loss_mask)
        else:
            self.policy_loss_mask.extend([policy_loss_mask] * len(ids))
        self.logprobs.extend(logprobs if logprobs else [0.0] * len(ids))
        if top_p_token_ids is not None and top_p_token_offsets is not None:
            self._extend_top_p_tokens(top_p_token_ids, top_p_token_offsets, expected_num_tokens=len(ids))
        elif self.top_p_token_offsets is not None:
            self.top_p_token_offsets.extend([self.top_p_token_offsets[-1]] * len(ids))
        self._pad_top_p_offsets_to_tokens()

    @staticmethod
    def _masked_logprobs(
        logprobs: list[float],
        loss_mask: int | list[int],
        token_count: int,
    ) -> list[float]:
        if not logprobs:
            return [0.0] * token_count
        values = list(logprobs)
        if len(values) < token_count:
            values.extend([0.0] * (token_count - len(values)))
        else:
            values = values[:token_count]
        if isinstance(loss_mask, list):
            return [float(value) if int(mask) == 1 else 0.0 for value, mask in zip(values, loss_mask, strict=True)]
        if int(loss_mask) == 0:
            return [0.0] * token_count
        return [float(value) for value in values]

    def _extend_top_p_tokens(
        self,
        token_ids: list[int],
        offsets: list[int],
        *,
        expected_num_tokens: int,
    ) -> None:
        assert (
            len(offsets) == expected_num_tokens + 1
        ), f"top-p token offsets length {len(offsets)} != generated token count + 1 {expected_num_tokens + 1}"
        assert offsets and offsets[0] == 0, f"top-p token offsets must start with 0, got {offsets[:1]}"
        assert offsets[-1] == len(
            token_ids
        ), f"top-p token offsets[-1] {offsets[-1]} != token ids length {len(token_ids)}"
        if self.top_p_token_ids is None:
            self.top_p_token_ids = []
            prefix_len = len(self.tokens) - expected_num_tokens
            self.top_p_token_offsets = [0] * (prefix_len + 1)
        assert self.top_p_token_offsets is not None
        base_offset = self.top_p_token_offsets[-1]
        self.top_p_token_ids.extend(token_ids)
        self.top_p_token_offsets.extend(base_offset + offset for offset in offsets[1:])

    def _top_p_slice(self, start: int, end: int) -> tuple[list[int], list[int]] | None:
        if self.top_p_token_ids is None or self.top_p_token_offsets is None:
            return None
        self._pad_top_p_offsets_to_tokens()
        start_offset = self.top_p_token_offsets[start]
        end_offset = self.top_p_token_offsets[end]
        token_ids = self.top_p_token_ids[start_offset:end_offset]
        offsets = [offset - start_offset for offset in self.top_p_token_offsets[start : end + 1]]
        return token_ids, offsets

    def _pad_top_p_offsets_to_tokens(self) -> None:
        if self.top_p_token_offsets is None:
            return
        target_len = len(self.tokens) + 1
        if len(self.top_p_token_offsets) < target_len:
            self.top_p_token_offsets.extend(
                [self.top_p_token_offsets[-1]] * (target_len - len(self.top_p_token_offsets))
            )

    def _truncate_top_p_tokens(self, length: int) -> None:
        if self.top_p_token_ids is None or self.top_p_token_offsets is None:
            return
        kept_token_count = self.top_p_token_offsets[length]
        del self.top_p_token_ids[kept_token_count:]
        del self.top_p_token_offsets[length + 1 :]

    def has_trained_response(self) -> bool:
        return any(self.loss_mask[self.leading_prompt_len :])

    def to_sample(
        self, base_sample: Sample, extra_metadata: dict[str, Any] | None, max_sample_tokens: int = 0
    ) -> Sample:
        """Emit the accumulated tokens as one ``Sample``, stripping the first-turn
        prompt so loss_mask / logprobs cover only the response region."""
        start = self.leading_prompt_len  # first-turn prompt stripped; response region starts here
        tokens = list(self.tokens)
        loss_mask = list(self.loss_mask)
        policy_loss_mask = list(self.policy_loss_mask)
        logprobs = list(self.logprobs)
        if max_sample_tokens and len(tokens) > max_sample_tokens:
            tokens = tokens[:max_sample_tokens]
            loss_mask = loss_mask[:max_sample_tokens]
            policy_loss_mask = policy_loss_mask[:max_sample_tokens]
            logprobs = logprobs[:max_sample_tokens]
            self._truncate_top_p_tokens(max_sample_tokens)
        md = dict(extra_metadata or {})
        turn_spans = []
        for span in self.turn_spans:
            response_start = int(span["response_token_start"])
            if response_start >= len(tokens):
                continue
            response_end = min(int(span["response_token_end"]), len(tokens))
            turn_spans.append(
                {
                    **span,
                    "response_token_end": response_end,
                    "truncated": response_end < int(span["response_token_end"]),
                }
            )
        versions = sorted(self.weight_versions)
        md.update(
            {
                "rollout_weight_versions": versions,
                "rollout_weight_version_count": len(versions),
                "rollout_trainable_tokens": int(sum(loss_mask[start:])),
                "rollout_logprob_invalid_tokens": 0,
                "tito_context_reasons": list(self.tito_context_reasons),
                "tito_exact_prefix_turns": sum(
                    reason in {"initial", "append_delta"} for reason in self.tito_context_reasons
                ),
                "tito_boundary_reason": self.boundary_reason,
                "tito_model_types": sorted(self.tito_model_types),
                "turn_spans": turn_spans,
            }
        )
        if len(versions) == 1:
            md["rollout_weight_version"] = versions[0]
        sample = Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=tokens,
            response_length=len(loss_mask) - start,
            loss_mask=loss_mask[start:],
            rollout_log_probs=logprobs[start:],
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata=md,
        )
        policy_loss_mask = policy_loss_mask[start:]
        if policy_loss_mask != sample.loss_mask:
            sample.policy_loss_mask = policy_loss_mask
        top_p_data = self._top_p_slice(start, len(tokens))
        if top_p_data is not None:
            sample.rollout_top_p_token_ids, sample.rollout_top_p_token_offsets = top_p_data
        return sample


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    def __init__(
        self,
        *,
        fork_threshold_tokens: int | None = None,
        strict_append_only: bool = False,
    ) -> None:
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._strict_append_only = strict_append_only
        self._trees: dict[str, MessageNode] = {}
        self._append_only_tails: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}
        self._weight_versions: dict[str, str] = {}

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        assert not turn.output_log_probs or len(turn.output_log_probs) == len(turn.output_ids), (
            f"turn.output_log_probs length {len(turn.output_log_probs)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )
        assert turn.loss_mask is None or len(turn.loss_mask) == len(turn.output_ids), (
            f"turn.loss_mask length {len(turn.loss_mask)} != " f"turn.output_ids length {len(turn.output_ids)}"
        )
        assert turn.policy_loss_mask is None or len(turn.policy_loss_mask) == len(turn.output_ids), (
            f"turn.policy_loss_mask length {len(turn.policy_loss_mask)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )
        if turn.context_delta_ids is not None:
            assert len(turn.context_delta_ids) <= len(turn.prompt_ids), (
                f"turn.context_delta_ids length {len(turn.context_delta_ids)} exceeds "
                f"turn.prompt_ids length {len(turn.prompt_ids)}"
            )
        assert (turn.rollout_top_p_token_ids is None) == (
            turn.rollout_top_p_token_offsets is None
        ), "turn.rollout_top_p_token_ids and turn.rollout_top_p_token_offsets must be set together"
        if turn.rollout_top_p_token_offsets is not None:
            assert len(turn.rollout_top_p_token_offsets) == len(turn.output_ids) + 1, (
                f"turn.rollout_top_p_token_offsets length {len(turn.rollout_top_p_token_offsets)} != "
                f"turn.output_ids length + 1 {len(turn.output_ids) + 1}"
            )

        train_mask = turn.loss_mask if turn.loss_mask is not None else [1] * len(turn.output_ids)
        has_trainable_tokens = (
            any(train_mask) if isinstance(train_mask, list) else bool(train_mask and turn.output_ids)
        )
        if turn.require_rollout_logprobs and has_trainable_tokens:
            if len(turn.output_log_probs) != len(turn.output_ids):
                raise ValueError(
                    "trainable rollout tokens require one finite logprob per output token: "
                    f"got {len(turn.output_log_probs)} logprobs for {len(turn.output_ids)} tokens"
                )
            invalid_indices = [
                index
                for index, (value, mask) in enumerate(zip(turn.output_log_probs, train_mask, strict=True))
                if mask and not math.isfinite(float(value))
            ]
            if invalid_indices:
                raise ValueError(
                    "trainable rollout tokens contain non-finite logprobs at indices " f"{invalid_indices[:8]}"
                )

        if turn.require_weight_version and turn.weight_version is None:
            raise ValueError("strict rollout versioning requires SGLang to return weight_version")
        if turn.weight_version is not None:
            observed_version = str(turn.weight_version)
            expected_version = self._weight_versions.setdefault(sid, observed_version)
            if observed_version != expected_version:
                raise ValueError(
                    f"mixed rollout weight versions for session {sid!r}: "
                    f"expected {expected_version!r}, observed {observed_version!r}"
                )

        root = self._trees.setdefault(sid, MessageNode())

        if self._strict_append_only:
            if turn.context_delta_ids is None:
                raise ValueError("strict append-only trajectories require context_delta_ids on every turn")
            if turn.tito_boundary_before:
                raise ValueError("strict append-only trajectories cannot contain a TiTO boundary")
            node = self._append_only_tails.get(sid, root)
            self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)
            self._append_only_tails[sid] = node.children[-1]
            return

        node, depth = self._find_mount_point(root, prompt_messages)
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth, incoming_turn=turn)
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        allow_fully_masked: bool = False,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:
        """Linearize this sid's routing tree into slime ``Sample`` objects and
        consume the session.

        Each routing leaf yields one or more Samples; ``reward`` is split evenly
        across all of them. The sid is dropped afterwards, so a second call for
        the same sid returns ``[]``.
        """
        root = self._trees.get(sid)
        if root is None:
            return []

        samples: list[Sample] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(
                self._chain_to_samples(
                    chain,
                    base_sample=base_sample,
                    extra_metadata=extra_metadata,
                    allow_fully_masked=allow_fully_masked,
                    max_sample_tokens=max_sample_tokens,
                )
            )

        per_sample_reward = (reward / len(samples)) if samples else 0.0
        for s in samples:
            s.reward = per_sample_reward

        self._trees.pop(sid, None)
        self._append_only_tails.pop(sid, None)
        self._turn_count.pop(sid, None)
        self._weight_versions.pop(sid, None)
        return samples

    def drop_session(self, sid: str) -> None:
        self._trees.pop(sid, None)
        self._append_only_tails.pop(sid, None)
        self._turn_count.pop(sid, None)
        self._weight_versions.pop(sid, None)

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
        """Walk down the tree matching each message by role and dict equality (==),
        returning the deepest node that still matches and where to mount the rest."""
        node = root
        depth = 0
        while depth < len(messages):
            msg = messages[depth]
            next_child = None
            for child in node.children:
                if child.role == msg.get("role") and child.message == msg:
                    next_child = child
                    break
            if next_child is None:
                break
            node = next_child
            depth += 1
        return node, depth

    def _try_merge_assistant_rewrite(
        self,
        sid: str,
        node: MessageNode,
        prompt_messages: list[dict[str, Any]],
        depth: int,
        incoming_turn: TurnRecord | None = None,
    ) -> tuple[MessageNode, int]:
        """Merge a short assistant-rewrite onto its node instead of forking.

        A harness may replay a prior assistant message re-rendered (e.g.
        whitespace, or Gemma4's structured tool_calls/tool_responses form) in a
        later prompt. It no longer matches the node we generated, so it would
        fork -- stranding the original generated turn as a dead-end leaf that
        still emits its own training Sample. Instead we swap the node's message
        for the rewrite so routing follows the live branch.

        What happens to the node's generated turn depends on what we can prove:

        * **strict TITO incoming turn** (``context_delta_ids`` set): the adapter
          computed that delta against its running accumulator, which contains
          this node's RAW output ids -- so the rewrite is purely a message-level
          rename and the turn keeps training in place. Nulling the turn here
          would leave the incoming delta dangling: the builder would drop this
          turn's tokens while every later delta still assumes they precede it,
          silently truncating the training context (and with it every
          intermediate action token).
        * **legacy incoming turn** (prefix-extension mode): the next prompt
          re-supplies the full context including the rewrite text, so the raw
          turn must vanish or its drifted tokens would force a fork. Demote to
          routing-only (``turn = None``), exactly the original behavior.

        This only applies below ``fork_threshold`` and when the mount point has
        exactly one assistant child that is a generated leaf; anything else
        forks, which is always safe (a rewrite mounts as routing-only).
        """
        if self._fork_threshold <= 0:
            return node, depth  # feature off
        if depth >= len(prompt_messages) or prompt_messages[depth].get("role") != "assistant":
            return node, depth  # genuine non-assistant history fork -> leave it

        asst_children = [c for c in node.children if c.role == "assistant"]
        if len(asst_children) != 1:
            if len(asst_children) > 1:
                logger.warning(
                    "record_turn(sid=%s turn=%s): %d assistant children at mount "
                    "point; can't tell which the rewrite targets, so forking.",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(asst_children),
                )
            return node, depth

        rewritten_node = asst_children[0]
        if (
            rewritten_node.children
            or rewritten_node.turn is None
            or len(rewritten_node.turn.output_ids) >= self._fork_threshold
        ):
            return node, depth

        retains_training = incoming_turn is not None and incoming_turn.context_delta_ids is not None
        rewritten_node.metadata["merged_rewrite"] = {
            "abandoned_turn_index": rewritten_node.turn_index,
            "abandoned_response_tokens": len(rewritten_node.turn.output_ids),
            "retains_training": retains_training,
        }
        if not retains_training:
            # Legacy prefix-extension mode: abandon the generated turn so the
            # rewrite text (re-tokenized in the next prompt) replaces it.
            rewritten_node.turn = None
            rewritten_node.turn_index = None
        rewritten_node.message = prompt_messages[depth]
        return rewritten_node, depth + 1

    def _mount_prompt_messages(
        self,
        node: MessageNode,
        remaining_messages: list[dict[str, Any]],
    ) -> MessageNode:
        for m in remaining_messages:
            node = node.add_child(MessageNode(role=m.get("role"), message=m))
        return node

    def _attach_assistant_leaf(
        self,
        sid: str,
        node: MessageNode,
        *,
        turn: TurnRecord,
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        asst = MessageNode(
            role="assistant",
            message=response_message,
            metadata=dict(metadata or {}),
        )
        asst.turn = turn
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        node.add_child(asst)
        self._turn_count[sid] = asst.turn_index

    def _split_chain_into_builders(self, chain: list[MessageNode]) -> list[_SampleBuilder]:
        """Pack the chain's generated turns into per-Sample token builders.

        Turns flow into the current builder until one can't extend it as an
        exact prefix (re-tokenization drift past what we can drop); that turn
        opens a new builder -- a fork. A generated turn shared by sibling leaves
        is trained only on the first leaf to claim it; later leaves re-emit it
        as loss_mask=0 context so the shared prefix isn't double-counted.
        """
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]

        builders: list[_SampleBuilder] = []
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True

            if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
                boundary_reason = asst_node.turn.tito_context_reason if builders else None
                builders.append(_SampleBuilder(self._fork_threshold, boundary_reason=boundary_reason))
                assert asst_node.turn_index is not None
                builders[-1].append_turn(
                    asst_node.turn,
                    DriftKind.CLEAN,
                    turn_index=asst_node.turn_index,
                    trained=trained,
                )
            else:
                assert asst_node.turn_index is not None
                builders[-1].append_turn(
                    asst_node.turn,
                    kind,
                    turn_index=asst_node.turn_index,
                    trained=trained,
                )
        return builders

    def _chain_to_samples(
        self,
        chain: list[MessageNode],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        allow_fully_masked: bool = False,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
        truncated = bool(asst_nodes) and asst_nodes[-1].turn.finish_reason == "length"
        use_tool = any(bool((n.message or {}).get("tool_calls")) for n in asst_nodes)
        ill_formed = any(n.turn.ill_formed for n in asst_nodes)
        md = {
            **(extra_metadata or {}),
            "truncated": truncated,
            "use_tool": use_tool,
            "ill_formed": ill_formed,
        }
        return [
            builder.to_sample(base_sample, md, max_sample_tokens)
            for builder in self._split_chain_into_builders(chain)
            if allow_fully_masked or builder.has_trained_response()
        ]


__all__ = [
    "TrajectoryManager",
    "TurnRecord",
]
