from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    action: Any = None


@dataclass
class ModelOutput:
    text: str = ""
    content: str = ""
    reasoning: str = ""
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"
    weight_version: str | None = None


@dataclass
class Step:
    observation: Any = None
    thought: str = ""
    action: Any = None
    model_response: str = ""
    chat_completions: list[dict[str, Any]] = field(default_factory=list)
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    weight_version: str | None = None


@dataclass
class Trajectory:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "fused_agent"
    task: Any = None
    steps: list[Step] = field(default_factory=list)
    reward: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    id: str = ""
    task: Any = None
    trajectories: list[Trajectory] = field(default_factory=list)
    is_correct: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

