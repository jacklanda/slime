# True On-Policy Training

True on-policy mode makes the sampled-token log probabilities produced by SGLang bitwise equal to those produced by Megatron before the policy update. It is stricter than experiment reproducibility: every numerical operation on the rollout and training paths must share the same contract.

## Qwen3 launchers

Both synchronous Odyssey launchers enable True On-Policy by default:

```bash
bash experiments/train_odyssey_qwen3_sync.sh
bash experiments/train_odyssey_qwen3_multinode_sync.sh
```

`--true-on-policy` remains the single public True On-Policy flag and can be passed explicitly to record intent. The default or the flag enables the complete internal contract. Do not add the internal flags individually. Set `TRUE_ON_POLICY=false` explicitly to run the legacy non-parity path.

- SGLang deterministic inference and its Qwen3 Megatron on-policy target
- FlashAttention 3 with a fixed attention split
- batch-invariant matrix multiplication, RMSNorm, and log-softmax kernels
- Megatron deterministic and batch-invariant modes with the Qwen3 residual contract
- unfused RoPE and SwiGLU paths where fusion changes operation ordering
- SGLang clean-prefill recomputation of accepted rollout log probabilities
- BF16-preserving rollout logprob transfer into training
- deterministic NCCL, Transformer Engine, and cuBLAS settings

The current contract supports dense Qwen3 models in BF16 and synchronous rollout. Partial and fully asynchronous rollout are rejected because their samples can cross policy versions.
Training and rollout tensor parallel sizes must match with the installed compatibility contract. Each launcher derives the rollout TP from the actor TP unless `ROLLOUT_NUM_GPUS_PER_ENGINE` is explicitly set, in which case a mismatch is rejected. Each rollout must feed exactly one optimizer step and each step must be published before the next rollout. Top-p, top-k, min-p, presence penalty, and repetition penalty must remain at their unmodified defaults because the training forward pass cannot reproduce those logit processors.

## Expected result

Monitor `train/train_rollout_logprob_abs_diff`. It must be exactly `0`, not merely close to zero. A nonzero value means that the installed SGLang, Megatron-LM, FlashAttention 3, DeepGEMM, or batch-invariant kernels do not implement the same numerical contract.

The prefill rescore flushes the SGLang radix cache before each scoring group. With consistent-hashing routing it scores samples separately on their original engine. This is intentionally slower, but prevents decode batching, prefix-cache state, and request scheduling from changing the reference log probabilities.

## Implementation

Bitwise parity depends on all of the following:

- **Attention:** training and inference use FlashAttention 3. Its deterministic split makes prefill and decode attention bitwise stable.
- **GEMM:** SGLang's batch-invariant implementation uses DeepGEMM when supported and a deterministic fallback otherwise.
- **Batch-invariant operators:** matrix multiplication, RMSNorm, reductions, and log-softmax use kernels derived from Thinking Machines Lab's `batch_invariant_ops`.
- **Operation ordering and dtype:** Qwen3 residuals, Q/K normalization, RoPE, SwiGLU, logits, and sampled-token log probabilities retain the same precision and operation order on both sides.
- **Scoring path:** accepted responses are scored again as clean prefills, and token IDs are checked before the original decode log probabilities are replaced.

This feature requires the SGLang and Megatron-LM revisions shipped with the slime environment. Merely enabling deterministic inference is not sufficient for true on-policy parity.
