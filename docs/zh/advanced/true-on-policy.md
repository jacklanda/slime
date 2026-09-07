# True On-Policy 训练

True on-policy 模式要求 SGLang 生成的 sampled-token log probability 与策略更新前 Megatron 计算的结果逐 bit 相同。它比实验可复现性更严格，因为 rollout 与 training 路径上的每一个数值操作都必须遵循同一契约。

## Qwen3 入口

单机和多节点同步 Odyssey 训练脚本均默认开启 True On-Policy：

```bash
bash experiments/train_odyssey_qwen3_sync.sh
bash experiments/train_odyssey_qwen3_multinode_sync.sh
```

`--true-on-policy` 是唯一公开的 True On-Policy 参数，也可以显式传入以记录运行意图。默认值或该参数都会统一启用 SGLang deterministic inference、FA3、Qwen3 Megatron on-policy contract、batch-invariant kernels、Megatron deterministic mode、必要的非融合算子、clean-prefill logprob 重算，以及 BF16 logprob 传输。不要单独拼装这些内部参数。如需运行旧的非 parity 路径，可显式设置环境变量 `TRUE_ON_POLICY=false`。

当前契约支持 BF16 dense Qwen3 模型和同步 rollout。partial rollout 与 fully-async rollout 会被拒绝，因为样本可能跨越不同策略版本。
当前兼容契约还要求 training TP 与 rollout TP 相同；两份脚本都会默认根据 actor TP 自动派生 rollout TP，若显式设置了不一致的 `ROLLOUT_NUM_GPUS_PER_ENGINE` 则会拒绝启动。每次 rollout 必须只对应一个 optimizer step，并在下一轮 rollout 前发布新权重。top-p、top-k、min-p、presence penalty 和 repetition penalty 必须保持不修改 logits 的默认值，因为训练 forward 无法复现这些 logit processor。

## 验证结果

监控 `train/train_rollout_logprob_abs_diff`，其值必须严格等于 `0`，而不是近似为零。非零结果表示当前 SGLang、Megatron-LM、FlashAttention 3、DeepGEMM 或 batch-invariant kernel 没有实现同一数值契约。

为了排除 decode batching、prefix cache 和请求调度的影响，系统会在训练前对 accepted responses 做 clean-prefill scoring，并检查返回 token ID 与原 response 完全一致。consistent-hashing 路由下会在原 engine 上逐样本重算，因此会牺牲一部分 rollout 性能。

## 实现要点

- Attention：训练与推理统一使用 FlashAttention 3 和确定性 attention split。
- GEMM：优先使用支持 batch invariance 的 DeepGEMM，否则使用确定性 fallback。
- Batch invariance：矩阵乘、RMSNorm、reduction 和 log-softmax 使用 batch-invariant kernels。
- 数值细节：Qwen3 residual、Q/K norm、RoPE、SwiGLU、logits 和 logprob 在两侧保持相同 dtype 与操作顺序。
- Scoring：accepted response 通过 clean prefill 重算，token 对齐校验成功后才覆盖 decode logprob。

该功能依赖 slime 环境中配套的 SGLang 与 Megatron-LM 版本。仅开启 deterministic inference 不能保证 true on-policy parity。
