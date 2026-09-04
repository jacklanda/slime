import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


NUM_GPUS = 0


@pytest.mark.unit
def test_eval_launcher_routes_bfcl_v3_to_gorilla_pipeline():
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (repo_root / "experiments/evals.sh").read_text(encoding="utf-8")
    bfcl_launcher = (repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard" / "run_eval.sh").read_text(encoding="utf-8")
    model_config = (repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard" / "bfcl_eval/constants/model_config.py").read_text(encoding="utf-8")

    assert "BFCL_BENCH_VERSION=v3" in launcher
    assert "BFCL_BENCH_VERSION=v4" in launcher
    assert 'BFCL_ROOT="${BENCHMARKS_ROOT}/gorilla/berkeley-function-call-leaderboard"' in launcher
    assert '--bench-version "${BFCL_BENCH_VERSION}"' in launcher
    assert '--local-model-path "$(realpath "${MODEL_DIR}")"' in launcher
    assert '--tp-size "${ROLLOUT_NUM_GPUS_PER_ENGINE}"' in launcher
    assert '--dp-size "${BFCL_DP_SIZE}"' in launcher
    assert 'BFCL_STICKY_ENGINE_ROUTING="${BFCL_STICKY_ENGINE_ROUTING:-true}"' in launcher
    assert "BFCL_NUM_THREADS=$((BFCL_DP_SIZE * 32))" in launcher
    assert '--max-running-requests "${BFCL_SGLANG_MAX_RUNNING_REQUESTS}"' in launcher
    assert "BFCL_CMD+=(--sticky-engine-routing)" in launcher
    assert '--agent-mode "${BFCL_AGENT_MODE}"' in launcher
    assert "BFCL_AGENT_MODE=slime_fused_gem" in launcher
    assert '--artifact-name "${EXPERIMENT_NAME}"' in launcher
    assert '"${BFCL_VENV_DIR}/bin/bfcl" evaluate' in launcher
    assert '"${BFCL_PYTHON_BIN}" -m venv --clear --system-site-packages' in launcher
    assert '"${BFCL_VENV_DIR}/bin/python" -m pip install --no-deps -e "${BFCL_ROOT}"' in launcher
    assert 'if [ "${BFCL_BENCH_VERSION}" = v4 ]' in launcher
    assert 'start_managed_serper "${BFCL_LOG_ROOT}/serper_search_server.log"' in launcher
    assert "SERPAPI_API_KEY" not in launcher
    assert "SERPAPI_API_KEY" not in bfcl_launcher
    assert "RETRIEVAL_SERVER_URL is required" in bfcl_launcher
    assert 'web_search_max_agent_steps=${BFCL_WEB_SEARCH_MAX_STEPS:-16}' in bfcl_launcher
    assert 'model_handler_utils=${PROJECT_ROOT}/bfcl_eval/model_handler/utils.py' in bfcl_launcher
    assert 'default_prompts=${PROJECT_ROOT}/bfcl_eval/constants/default_prompts.py' in bfcl_launcher
    assert "'sentence-transformers==3.4.1'" in launcher
    assert '"${BFCL_VENV_DIR}/bin/python" -m pip install --no-deps' in launcher
    assert '--sglang-python-bin "${BFCL_SGLANG_PYTHON_BIN}"' in launcher
    assert 'export BFCL_SGLANG_LD_PRELOAD="${BFCL_SGLANG_LIBSTDCXX}"' in launcher
    assert "export BFCL_DISCARD_HISTORICAL_THINKING=" in launcher
    assert 'export BFCL_VERSION_PREFIX="BFCL_${BFCL_BENCH_VERSION}"' in launcher
    assert 'mkdir -p "${BFCL_RESULT_DIR}" "${BFCL_SCORE_DIR}"' in launcher
    assert "export BFCL_DATA_DIR" in launcher
    assert '--bfcl-v3-commit "${BFCL_V3_COMMIT}"' in launcher
    assert '--bfcl-data-dir "${BFCL_DATA_DIR}"' in launcher
    assert "unset BFCL_DATA_DIR || true" in launcher
    assert '*qwen3.5-4b*) BFCL_MODEL_KEY="Qwen/Qwen3.5-4B"' in launcher
    assert '*qwen3-8b*) BFCL_MODEL_KEY="Qwen/Qwen3-8B"' in launcher
    assert '*qwen3-14b*) BFCL_MODEL_KEY="Qwen/Qwen3-14B"' in launcher
    assert "export BFCL_SLIME_TOOL_PARSER_PATH=" in launcher
    assert 'export FUSED_MODEL_SERIES="${MODEL_SERIES}"' in launcher
    assert 'export FUSED_MAX_STEPS="${MAX_STEPS}"' in launcher
    assert 'export FUSED_MCP_MAX_STEPS="${MCP_MAX_STEPS}"' in launcher
    assert 'BFCL_SEED="${BFCL_SEED:-${ROLLOUT_SEED}}"' in launcher
    assert 'BFCL_MAX_TOKENS="${EVAL_MAX_RESPONSE_LEN}"' in launcher
    assert 'export BFCL_WEB_SEARCH_MAX_STEPS="${BFCL_WEB_SEARCH_MAX_STEPS}"' in launcher
    norm_fallback = 'export FLASHINFER_USE_TORCH_NORM="${FLASHINFER_USE_TORCH_NORM:-1}"'
    assert launcher.index(norm_fallback) < launcher.index('"${BFCL_CMD[@]}"')
    assert "BFCL_WEB_SEARCH_MAX_STEPS=16" in launcher
    assert 'BFCL_DISCARD_HISTORICAL_THINKING="${BFCL_DISCARD_HISTORICAL_THINKING:-}"' in launcher
    assert 'BFCL_DISCARD_HISTORICAL_THINKING="${DISCARD_HISTORICAL_THINKING}"' in launcher
    assert 'bfcl_discard_historical_thinking_explicit=true' in launcher
    assert 'BFCL_CMD+=(--seed "${BFCL_SEED}")' in launcher
    assert "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" in launcher
    assert "BFCL_SGLANG_JSON_MODEL_OVERRIDE_ARGS=" in launcher
    assert "slime_plugins.evals.bfcl_reliability_report" in launcher
    assert '"Qwen/Qwen3-4B-FC"' in model_config
    assert '"Qwen/Qwen3-4B-FC-RLLM-ToolAgent"' in model_config
    assert '"Qwen/Qwen3-4B"' in model_config
    assert '"Qwen/Qwen3-4B-Thinking-2507-FC-RLLM-ToolAgent"' in model_config
    assert '"Qwen/Qwen3.5-4B-FC-RLLM-ToolAgent"' in model_config
    assert '"Qwen/Qwen3-8B-FC-RLLM-ToolAgent"' in model_config
    assert '"Qwen/Qwen3-14B-FC-RLLM-ToolAgent"' in model_config


@pytest.mark.unit
def test_bfcl_tool_agent_discards_only_historical_reasoning(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    compat_path = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard" / "bfcl_eval/model_handler/tool_agent_compat.py"
    spec = importlib.util.spec_from_file_location("bfcl_tool_agent_compat_test", compat_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    agent = module.MCPToolAgentCompat()
    agent.messages = [
        {"role": "system", "content": "tools"},
        {"role": "assistant", "content": "call", "reasoning_content": "old think"},
        {"role": "tool", "content": "ok"},
    ]
    monkeypatch.setenv("BFCL_DISCARD_HISTORICAL_THINKING", "true")

    request_messages = agent.chat_completions
    assert "reasoning_content" not in request_messages[1]
    assert agent.messages[1]["reasoning_content"] == "old think"

    monkeypatch.setenv("BFCL_DISCARD_HISTORICAL_THINKING", "false")
    agent.discard_historical_thinking = True
    request_messages = agent.chat_completions
    assert "reasoning_content" not in request_messages[1]
    assert agent.messages[1]["reasoning_content"] == "old think"


@pytest.mark.unit
def test_bfcl_memory_prompts_require_retention_and_retrieval():
    repo_root = Path(__file__).resolve().parents[1]
    prompt_source = (
        repo_root
        / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
        / "bfcl_eval/constants/default_prompts.py"
    ).read_text(encoding="utf-8")
    utils_source = (
        repo_root
        / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
        / "bfcl_eval/model_handler/utils.py"
    ).read_text(encoding="utf-8")

    assert "store every concrete fact that may be asked about later" in prompt_source
    assert "Use archival memory for overflow" in prompt_source
    assert "Before saying that information is unknown" in prompt_source
    assert 'if "prereq" in test_category:' in utils_source


@pytest.mark.unit
def test_bfcl_tool_agent_uses_slime_gem_prompt_and_finish_schema():
    repo_root = Path(__file__).resolve().parents[1]
    compat_path = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard" / "bfcl_eval/model_handler/tool_agent_compat.py"
    spec = importlib.util.spec_from_file_location("bfcl_tool_agent_prompt_test", compat_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    agent = module.MCPToolAgentCompat(
        tools_json=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
    )
    system_prompt = agent.messages[0]["content"]
    assert "Use available non-finish tools when they are relevant" in system_prompt
    assert "do not call an unrelated tool" in system_prompt
    assert '"name": "lookup"' in system_prompt
    assert '"name": "finish"' in system_prompt


@pytest.mark.unit
def test_bfcl_tool_agent_preserves_final_response_after_actions():
    repo_root = Path(__file__).resolve().parents[1]
    gorilla_root = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
    compat_path = gorilla_root / "bfcl_eval/model_handler/tool_agent_compat.py"
    handler_path = gorilla_root / "bfcl_eval/model_handler/local_inference/rllm_qwen_tool_agent.py"
    checker_path = gorilla_root / "bfcl_eval/eval_checker/agentic_eval/agentic_checker.py"
    spec = importlib.util.spec_from_file_location("bfcl_tool_agent_actions_test", compat_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    checker_spec = importlib.util.spec_from_file_location(
        "bfcl_agentic_checker_test", checker_path
    )
    checker = importlib.util.module_from_spec(checker_spec)
    assert checker_spec.loader is not None
    checker_spec.loader.exec_module(checker)

    actions = [
        {"function": {"name": "lookup", "arguments": {"id": 1}}},
        {"function": {"name": "update", "arguments": {"id": 1}}},
        {
            "function": {
                "name": "finish",
                "arguments": json.dumps(
                    {"command": "submit", "result": ["final answer"]}
                ),
            }
        },
        {"function": {"name": "ignored", "arguments": {}}},
    ]
    executable, final_response = module.split_actions_and_final_response(actions)

    assert [action["function"]["name"] for action in executable] == ["lookup", "update"]
    assert final_response == '["final answer"]'
    assert checker.agentic_checker(final_response, ["final answer"])["valid"] is True

    agent = module.MCPToolAgentCompat()
    plain_response = '{"answer": "Michael", "context": "Stored in memory."}'
    action = agent.update_from_model(plain_response)
    executable, final_response = module.split_actions_and_final_response(action.action)

    assert executable == []
    assert final_response == plain_response
    assert checker.agentic_checker(final_response, ["Michael"])["valid"] is True

    handler_source = handler_path.read_text(encoding="utf-8")
    assert "current_turn_response.append(final_response)" in handler_source
    assert handler_source.count("return convert_to_function_call(self._decode_ast_calls(calls))") == 2
    assert "execute_multi_turn_func_call(\n                    convert_to_function_call(executable_tool_call_maps)" in handler_source
    assert "Executed only the first parsed action" not in handler_source


@pytest.mark.unit
def test_bfcl_tool_agent_loads_current_slime_parser(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    gorilla_root = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
    compat_path = gorilla_root / "bfcl_eval/model_handler/tool_agent_compat.py"
    parser_path = repo_root / "slime/rollout/fused_agent/parser.py"
    monkeypatch.setenv("BFCL_SLIME_TOOL_PARSER_PATH", str(parser_path))

    spec = importlib.util.spec_from_file_location("bfcl_tool_agent_slime_parser_test", compat_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.get_tool_parser("qwen", valid_tools={"lookup", "finish"})

    assert isinstance(parser, module.SlimeQwenToolParserAdapter)
    calls = parser.parse('<tool_call>{"name":"lookup","arguments":{"id":1}}</tool_call>')
    assert [(call.name, call.arguments) for call in calls] == [("lookup", {"id": 1})]


@pytest.mark.unit
def test_bfcl_tool_agent_formats_tool_observations_like_slime(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    gorilla_root = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
    compat_path = gorilla_root / "bfcl_eval/model_handler/tool_agent_compat.py"
    parser_path = repo_root / "slime/rollout/fused_agent/parser.py"
    monkeypatch.setenv("BFCL_SLIME_TOOL_PARSER_PATH", str(parser_path))

    spec = importlib.util.spec_from_file_location("bfcl_tool_agent_observation_test", compat_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    agent = module.MCPToolAgentCompat()
    agent.update_from_env(
        observation={
            "tool_outputs": {
                "call-1": {"name": "lookup", "content": '{"value": 1}'},
            }
        },
        reward=0.0,
        done=False,
        info={},
    )

    assert agent.messages[-1] == {
        "role": "user",
        "content": ("<tool_response>\n" "Execution output of [lookup]:\n" '{"value": 1}\n' "</tool_response>"),
    }


@pytest.mark.unit
def test_bfcl_tool_agent_selects_qwen35_xml_parser(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    gorilla_root = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
    compat_path = gorilla_root / "bfcl_eval/model_handler/tool_agent_compat.py"
    parser_path = repo_root / "slime/rollout/fused_agent/parser.py"
    monkeypatch.setenv("BFCL_SLIME_TOOL_PARSER_PATH", str(parser_path))
    monkeypatch.setenv("FUSED_MODEL_SERIES", "qwen3.5")

    spec = importlib.util.spec_from_file_location("bfcl_tool_agent_qwen35_parser_test", compat_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.get_tool_parser("qwen", valid_tools={"lookup", "finish"})

    assert type(parser._parser).__name__ == "Qwen3CoderToolParser"
    calls = parser.parse("<tool_call>\n<function=lookup>\n<parameter=id>\n1\n</parameter>\n</function>\n</tool_call>")
    assert [(call.name, call.arguments) for call in calls] == [("lookup", {"id": "1"})]
    assert "<function=FUNCTION_NAME>" in parser.get_tool_prompt("")
    assert '"name": <function-name>' not in parser.get_tool_prompt("")


@pytest.mark.unit
def test_bfcl_slime_parser_source_load_is_thread_safe(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    compat_path = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard" / "bfcl_eval/model_handler/tool_agent_compat.py"
    parser_path = tmp_path / "slow_parser.py"
    parser_path.write_text(
        """\
import time
time.sleep(0.1)

class Parser:
    def __init__(self, valid_tools=None):
        self.valid_tools = valid_tools

def make_tool_parser(model_name, valid_tools=None):
    return Parser(valid_tools)
""",
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location("bfcl_tool_agent_parser_race_test", compat_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.delitem(sys.modules, "_bfcl_slime_fused_agent_parser", raising=False)

    def load_parser(_):
        return module.SlimeQwenToolParserAdapter(str(parser_path), valid_tools={"lookup"})

    with ThreadPoolExecutor(max_workers=16) as executor:
        parsers = list(executor.map(load_parser, range(16)))

    assert all(parser.valid_tools == {"lookup"} for parser in parsers)


@pytest.mark.unit
def test_bfcl_manifest_rejects_reuse_with_changed_settings(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard" / "bfcl_eval/run_manifest.py"
    spec = importlib.util.spec_from_file_location("bfcl_run_manifest_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    manifest_path = tmp_path / "run_manifest.json"
    module.validate_or_write_manifest(manifest_path, {"benchmark_version": "v3"}, False)
    assert json.loads(manifest_path.read_text())["benchmark_version"] == "v3"
    with pytest.raises(RuntimeError, match="different settings"):
        module.validate_or_write_manifest(manifest_path, {"benchmark_version": "v4"}, False)


@pytest.mark.unit
def test_bfcl_manifest_hashes_adapter_sources(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard" / "bfcl_eval/run_manifest.py"
    spec = importlib.util.spec_from_file_location("bfcl_run_manifest_hash_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    source = tmp_path / "adapter.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    first_hash = module.file_hashes([f"adapter={source}"])
    source.write_text("VERSION = 2\n", encoding="utf-8")
    second_hash = module.file_hashes([f"adapter={source}"])

    assert first_hash.keys() == {"adapter"}
    assert first_hash != second_hash

    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    (evaluator / "checker.py").write_text("VERSION = 1\n", encoding="utf-8")
    first_tree_hash = module.tree_hashes([f"evaluator={evaluator}"])
    (evaluator / "checker.py").write_text("VERSION = 2\n", encoding="utf-8")
    second_tree_hash = module.tree_hashes([f"evaluator={evaluator}"])
    assert first_tree_hash.keys() == {"evaluator"}
    assert first_tree_hash != second_tree_hash


@pytest.mark.unit
def test_bfcl_run_eval_forwards_disabled_top_k():
    repo_root = Path(__file__).resolve().parents[1]
    run_eval = (repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard/run_eval.sh").read_text(encoding="utf-8")

    assert 'GENERATE_CMD+=(--top-k "${TOP_K}")' in run_eval
    assert 'if [[ "${TOP_K}" != "-1" ]]' not in run_eval
    assert '--field "max_agent_steps=${FUSED_MCP_MAX_STEPS:-${FUSED_MAX_STEPS:-16}}"' in run_eval
    assert '--identity-file "tool_agent_handler=' in run_eval
    assert '--identity-file "tool_agent_compat=' in run_eval
    assert '--identity-file "base_oss_handler=' in run_eval
    assert '--identity-file "sticky_engine_routing=' in run_eval
    assert '--identity-file "slime_tool_parser=' in run_eval
    assert '--identity-tree "bfcl_evaluator=' in run_eval
    assert "GENERATE_CMD+=(--sticky-engine-routing)" in run_eval
    assert '--field "sticky_engine_routing=${STICKY_ENGINE_ROUTING}"' in run_eval
    assert '--field "engine_count=' in run_eval


@pytest.mark.unit
def test_bfcl_sticky_engine_routing_is_balanced_and_thread_local():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root
        / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
        / "bfcl_eval/model_handler/local_inference/sticky_engine_routing.py"
    )
    spec = importlib.util.spec_from_file_location("bfcl_sticky_engine_routing_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    router = module.TrajectoryEngineRouter()
    router.clients = [object() for _ in range(8)]
    default_client = object()
    bound = threading.Barrier(16)
    checked = threading.Barrier(16)

    def route_trajectory(trajectory_id):
        selected_index = router.bind(trajectory_id)
        bound.wait()
        selected_clients = [router.current_client(default_client) for _ in range(20)]
        checked.wait()
        router.unbind()
        return selected_index, selected_clients, router.current_client(default_client)

    trajectory_ids = [f"trajectory-{index}" for index in range(16)]
    with ThreadPoolExecutor(max_workers=16) as executor:
        routed = list(executor.map(route_trajectory, trajectory_ids))

    engine_indices = [engine_index for engine_index, _, _ in routed]
    assert [engine_indices.count(index) for index in range(8)] == [2] * 8
    for engine_index, selected_clients, unbound_client in routed:
        assert all(client is router.clients[engine_index] for client in selected_clients)
        assert unbound_client is default_client


@pytest.mark.unit
def test_bfcl_sticky_engine_startup_contract():
    repo_root = Path(__file__).resolve().parents[1]
    gorilla_root = (
        repo_root
        / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
    )
    handler = (
        gorilla_root
        / "bfcl_eval/model_handler/local_inference/base_oss_handler.py"
    ).read_text(encoding="utf-8")
    tool_agent_handler = (
        gorilla_root
        / "bfcl_eval/model_handler/local_inference/rllm_qwen_tool_agent.py"
    ).read_text(encoding="utf-8")
    cli = (gorilla_root / "bfcl_eval/__main__.py").read_text(encoding="utf-8")
    generation = (gorilla_root / "bfcl_eval/_llm_response_generation.py").read_text(
        encoding="utf-8"
    )

    assert 'engine_env["CUDA_VISIBLE_DEVICES"] = visible_devices[engine_index]' in handler
    assert 'common_cmd + ["--port", str(port)]' in handler
    assert '"--tp-size",\n                        "1"' in handler
    assert '"--dp-size",\n                        "1"' in handler
    assert "for engine_index, port in enumerate(ports):" in handler
    assert "self._engine_router.bind(test_entry[\"id\"])" in handler
    assert "self._engine_router.bind(test_entry[\"id\"])" in tool_agent_handler
    assert "self._rllm_agent" not in tool_agent_handler
    assert 'common_cmd += ["--max-running-requests", str(max_running_requests)]' in handler
    assert 'sglang_cmd += ["--max-running-requests", str(max_running_requests)]' in handler
    assert handler.count('"--json-model-override-args"') == 2
    assert 'sglang_model_override_args=${BFCL_SGLANG_JSON_MODEL_OVERRIDE_ARGS:-}' in (
        gorilla_root / "run_eval.sh"
    ).read_text(encoding="utf-8")
    assert '"--sticky-engine-routing"' in cli
    assert "sticky_engine_routing=sticky_engine_routing" in cli
    assert "sticky_engine_routing=args.sticky_engine_routing" in generation


@pytest.mark.unit
def test_bfcl_version_prefix_can_select_historical_v3(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    mapping_path = repo_root / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard" / "bfcl_eval/constants/category_mapping.py"
    monkeypatch.setenv("BFCL_VERSION_PREFIX", "BFCL_v3")
    spec = importlib.util.spec_from_file_location("bfcl_v3_category_mapping_test", mapping_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.VERSION_PREFIX == "BFCL_v3"


@pytest.mark.unit
def test_bfcl_reliability_report_flags_protocol_and_runtime_limits(tmp_path):
    from slime_plugins.evals.bfcl_reliability_report import build_report, write_report

    artifact = "model"
    result_root = tmp_path / "results" / artifact
    score_root = tmp_path / "scores" / artifact
    (result_root / "multi_turn").mkdir(parents=True)
    (result_root / "agentic").mkdir(parents=True)
    (score_root / "non_live").mkdir(parents=True)
    (score_root / "multi_turn").mkdir(parents=True)
    (score_root / "agentic").mkdir(parents=True)
    (result_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "seed": "42",
                "temperature": "0.6",
                "context_length": "40960",
                "max_tokens": "4096",
                "max_agent_steps": "128",
                "web_search_max_agent_steps": "16",
                "discard_historical_thinking": "true",
            }
        ),
        encoding="utf-8",
    )
    (score_root / "non_live/BFCL_v3_parallel_score.json").write_text(
        "\n".join(
            [
                json.dumps({"accuracy": 0.5, "correct_count": 1, "total_count": 2}),
                json.dumps(
                    {
                        "test_category": "parallel",
                        "valid": False,
                        "error_type": "parallel_function_checker_no_order:wrong_count",
                        "model_result_decoded": [{"lookup": {"id": 1}}],
                        "possible_answer": [{"lookup": {"id": [1]}}, {"lookup": {"id": [2]}}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (score_root / "multi_turn/BFCL_v3_multi_turn_base_score.json").write_text(
        "\n".join(
            [
                json.dumps({"accuracy": 0.0, "correct_count": 0, "total_count": 2}),
                json.dumps(
                    {
                        "test_category": "multi_turn_base",
                        "valid": False,
                        "error": {"error_type": "multi_turn:force_terminated"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (result_root / "multi_turn/BFCL_v3_multi_turn_base_result.json").write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "result": "Error during inference: No generation budget remains within the model context window."}),
                json.dumps({"id": "b", "result": [[]], "inference_log": "Model has been forced to quit after 128 total steps."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (score_root / "agentic/BFCL_v4_web_search_base_score.json").write_text(
        "\n".join(
            [
                json.dumps({"accuracy": 0.0, "correct_count": 0, "total_count": 1}),
                json.dumps(
                    {
                        "id": "web_search_base_0",
                        "test_category": "web_search_base",
                        "valid": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (result_root / "agentic/BFCL_v4_web_search_base_result.json").write_text(
        json.dumps(
            {
                "id": "web_search_base_0",
                "result": "Error during inference: No generation budget remains within the model context window.",
                "output_token_count": [[4096]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_report(tmp_path / "results", tmp_path / "scores", artifact)
    write_report(report, tmp_path / "scores")

    assert report["diagnostics"]["parallel_protocol_mismatch"] == {"count": 1, "total": 2, "rate": 0.5}
    assert report["diagnostics"]["multi_turn_context_exhaustion"] == {"count": 1, "total": 2, "rate": 0.5}
    assert report["diagnostics"]["multi_turn_step_limit_reached"] == {"count": 1, "total": 2, "rate": 0.5}
    assert report["diagnostics"]["multi_turn_force_terminated"] == {"count": 1, "total": 2, "rate": 0.5}
    assert report["diagnostics"]["agentic_context_exhaustion"] == {"count": 1, "total": 1, "rate": 1.0}
    assert report["diagnostics"]["failed_generation_token_cap_reached"] == {
        "count": 1,
        "total": 5,
        "rate": 0.2,
    }
    assert report["configuration"]["max_tokens"] == "4096"
    assert report["configuration"]["web_search_max_agent_steps"] == "16"
    assert report["category_uncertainty"][0]["wilson_95_low"] < 0.5
    assert (tmp_path / "scores/reliability_report.json").is_file()
    assert (tmp_path / "scores/data_category_uncertainty.csv").is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
