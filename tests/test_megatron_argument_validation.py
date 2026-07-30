import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


def load_arguments_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    arguments_mod = types.ModuleType("megatron.training.arguments")
    tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
    tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
    transformers_mod = types.ModuleType("transformers")

    arguments_mod.parse_args = lambda *args, **kwargs: None
    arguments_mod.validate_args = lambda args: args
    tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "arguments.py"
    module_name = "test_megatron_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_slime_arguments_module(monkeypatch):
    router_pkg_mod = types.ModuleType("sglang_router")
    router_launch_mod = types.ModuleType("sglang_router.launch_router")
    sglang_arguments_mod = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_external_mod = types.ModuleType("slime.backends.sglang_utils.external")
    logging_utils_mod = types.ModuleType("slime.utils.logging_utils")

    router_launch_mod.RouterArgs = object
    sglang_arguments_mod.sglang_parse_args = lambda *args, **kwargs: None
    sglang_arguments_mod.validate_args = lambda args: args
    sglang_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_arguments_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", sglang_external_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "utils" / "arguments.py"
    module_name = "test_slime_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_qwen3_6_args(**overrides):
    values = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_qwen3_6_hf_config():
    text_config = types.SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000},
    )
    return types.SimpleNamespace(text_config=text_config)


def make_allgather_cp_args(**overrides):
    values = dict(
        allgather_cp=True,
        context_parallel_size=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factor", "original_context", "message"),
    [
        (1.0, 32768, "scaling-factor"),
        (float("nan"), 32768, "scaling-factor"),
        (4.0, 0, "original-max-position-embeddings"),
    ],
)
def test_validate_args_rejects_invalid_yarn_settings(monkeypatch, factor, original_context, message):
    module = load_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        use_yarn_rope=True,
        yarn_rope_scaling_factor=factor,
        yarn_original_max_position_embeddings=original_context,
    )

    with pytest.raises(ValueError, match=message):
        module.validate_args(args)


@pytest.mark.unit
def test_apply_yarn_sglang_override_merges_structured_json(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        use_yarn_rope=True,
        yarn_rope_scaling_factor=4.0,
        yarn_original_max_position_embeddings=32768,
        sglang_context_length=131072,
        sglang_json_model_override_args='{"architectures":["Qwen3ForCausalLM"]}',
    )

    module._apply_yarn_sglang_override(args)

    assert args.sglang_json_model_override_args == (
        '{"architectures":["Qwen3ForCausalLM"],'
        '"max_position_embeddings":131072,'
        '"rope_scaling":{"rope_type":"yarn","factor":4.0,'
        '"original_max_position_embeddings":32768}}'
    )


@pytest.mark.unit
def test_apply_yarn_sglang_override_rejects_invalid_json(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        use_yarn_rope=True,
        yarn_rope_scaling_factor=4.0,
        yarn_original_max_position_embeddings=32768,
        sglang_context_length=131072,
        sglang_json_model_override_args='{"broken":',
    )

    with pytest.raises(ValueError, match="must be valid JSON"):
        module._apply_yarn_sglang_override(args)


@pytest.mark.unit
def test_hf_validate_all_moe_skips_dense_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._hf_validate_args(make_qwen3_6_args(), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_moe_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    with pytest.raises(AssertionError, match="moe_intermediate_size"):
        module._hf_validate_args(make_qwen3_6_args(moe_ffn_hidden_size=256), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_dense_intermediate_size_when_moe_has_dense_layers(monkeypatch):
    module = load_arguments_module(monkeypatch)

    args = make_qwen3_6_args(moe_layer_freq=[0] + [1] * 39)

    with pytest.raises(AssertionError, match="intermediate_size"):
        module._hf_validate_args(args, make_qwen3_6_hf_config())


@pytest.mark.unit
def test_allgather_cp_rejects_non_dsa_cp_models(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args()
    hf_config = types.SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3")

    with pytest.raises(ValueError, match="only supported for DSA attention models"):
        module._validate_allgather_cp_supported(args, hf_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hf_config",
    [
        types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v3"),
        types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="glm"),
    ],
)
def test_allgather_cp_allows_dsa_architectures(monkeypatch, hf_config):
    module = load_arguments_module(monkeypatch)

    module._validate_allgather_cp_supported(make_allgather_cp_args(), hf_config)


@pytest.mark.unit
def test_allgather_cp_ignores_cp_size_one(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args(context_parallel_size=1)

    module._validate_allgather_cp_supported(args)


@pytest.mark.unit
def test_variable_sequences_reject_allgather_moe_dispatcher(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        moe_token_dispatcher_type="allgather",
        pipeline_model_parallel_size=1,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
    )

    with pytest.raises(ValueError, match="use --moe-token-dispatcher-type alltoall"):
        module.validate_args(args)


@pytest.mark.unit
def test_variable_sequences_keep_explicit_alltoall_dispatcher(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        moe_token_dispatcher_type="alltoall",
        pipeline_model_parallel_size=1,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
    )

    module.validate_args(args)

    assert args.variable_seq_lengths is True
    assert args.moe_token_dispatcher_type == "alltoall"


@pytest.mark.unit
def test_update_weight_disk_dir_required_for_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(update_weight_transport="disk", update_weight_disk_dir=None)

    with pytest.raises(ValueError, match="update-weight-disk-dir"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weights_interval_must_be_positive(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(update_weights_interval=0)

    with pytest.raises(ValueError, match="update-weights-interval"):
        module.slime_validate_args(args)


def make_slime_validate_args(**overrides):
    values = dict(
        eval_config=None,
        eval_prompt_data=None,
        kl_coef=0,
        use_kl_loss=False,
        ref_load=None,
        use_opd=False,
        opd_type=None,
        opd_teacher_load=None,
        megatron_to_hf_mode="raw",
        load=None,
        hf_checkpoint="/tmp/hf",
        ref_ckpt_step=None,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=None,
        eval_interval=None,
        save_interval=None,
        save=None,
        kl_loss_coef=0,
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_max_tokens_per_gpu=None,
        balance_by_flops=False,
        balance_data=False,
        eps_clip_high=None,
        eps_clip=0.2,
        eval_reward_key=None,
        reward_key="reward",
        dump_details=None,
        save_debug_rollout_data=None,
        save_debug_train_data=None,
        load_debug_rollout_data=None,
        rollout_external_engine_addrs=None,
        debug_train_only=False,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        offload=False,
        offload_train=None,
        offload_rollout=None,
        debug_rollout_only=False,
        colocate=False,
        rollout_num_gpus=8,
        train_memory_margin_bytes=0,
        eval_function_path=None,
        rollout_function_path="custom.rollout",
        num_steps_per_rollout=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=None,
        grpo_std_normalization=True,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        rollout_global_dataset=False,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        custom_config_path=None,
        eval_max_context_len=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        train_backend="megatron",
        release_train=False,
        keep_old_actor=False,
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None,
        update_weight_mode="full",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_slime_validate_args_rejects_webqa_prompt_data_without_ground_truth(monkeypatch, tmp_path):
    import pandas as pd

    module = load_slime_arguments_module(monkeypatch)
    path = tmp_path / "webqa.parquet"
    pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "Q1"}],
                "reward_model": {"ground_truth": None, "style": "rule"},
                "extra_info": {"ground_truth": None, "question": "Q1"},
            },
            {
                "prompt": [{"role": "user", "content": "Q2"}],
                "reward_model": {"ground_truth": None, "style": "rule"},
                "extra_info": {"ground_truth": None, "question": "Q2"},
            },
        ]
    ).to_parquet(path)

    args = make_slime_validate_args(
        prompt_data=str(path),
        label_key="reward_model",
        metadata_key="extra_info",
        rollout_function_path="slime.rollout.fused_agent.generate.generate",
    )

    with pytest.raises(ValueError, match="sample check failed"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_slime_validate_args_allows_endless_terminal_fused_data_without_ground_truth(monkeypatch, tmp_path):
    import pandas as pd

    module = load_slime_arguments_module(monkeypatch)
    path = tmp_path / "terminal_tasks.parquet"
    pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "Fix the repository"}],
                "reward_model": {"ground_truth": None, "style": "rule"},
                "extra_info": {"data_source": "endless_terminals", "task_id": "task-1"},
            }
        ]
    ).to_parquet(path)

    args = make_slime_validate_args(
        prompt_data=str(path),
        label_key="reward_model",
        metadata_key="extra_info",
        custom_generate_function_path="slime.rollout.fused_agent.generate.generate",
    )

    module.slime_validate_args(args)


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=True, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0


@pytest.mark.unit
def test_slime_validate_args_preserves_larger_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        colocate=True,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        rollout_num_gpus=12,
    )

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 12


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_without_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=False, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.actor_num_gpus_per_node == 8
    assert args.actor_num_nodes == 1


@pytest.mark.unit
def test_update_weight_delta_requires_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="nccl",
        update_weight_local_checkpoint_dir="/local/ckpt",
    )

    with pytest.raises(ValueError, match="requires --update-weight-transport=disk"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir="/local/ckpt",
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_requires_local_checkpoint_dir(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir=None,
    )

    with pytest.raises(ValueError, match="requires --update-weight-local-checkpoint-dir"):
        module.slime_validate_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
