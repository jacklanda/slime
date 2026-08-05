MODEL_ARGS=(
    --spec "slime_plugins.models.gemma4" "get_gemma4_spec"
    --custom-model-provider-path "slime_plugins.models.gemma4_provider.model_provider"
    --num-layers 35
    --hidden-size 1536
    --ffn-hidden-size 6144
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 1
    --kv-channels 256
    --use-rotary-position-embeddings
    --disable-bias-linear
    --normalization "RMSNorm"
    --norm-epsilon 1e-6
    --rotary-base 10000
    --rotary-percent 1.0
    --vocab-size 262144
    --qk-layernorm
)
