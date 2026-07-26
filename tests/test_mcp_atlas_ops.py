import importlib.util
import json
import sys
from pathlib import Path


def _load_mutation_report():
    path = (
        Path(__file__).parents[1]
        / "experiments/artifacts/benchmarks/mcp-atlas/ops/mutation_report.py"
    )
    spec = importlib.util.spec_from_file_location("mcp_atlas_mutation_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def test_mcp_atlas_mutation_report_extracts_only_persistent_writes(tmp_path):
    module = _load_mutation_report()
    evals = tmp_path / "evals"
    evals.mkdir()
    payload = {
        "trajectories": [
            {
                "episode_id": "episode-1",
                "task": {"task_id": "task-1"},
                "trajectories": [
                    {
                        "steps": [
                            {
                                "action": '<tool_call>{"name":"slack_conversations_add_message",'
                                '"arguments":{"channel_id":"C1","text":"hello"}}</tool_call>'
                            },
                            {
                                "action": '<tool_call>{"name":"slack_channels_list",'
                                '"arguments":{}}</tool_call>'
                            },
                        ]
                    }
                ],
            }
        ]
    }
    (evals / "global_steps_0.json").write_text(json.dumps(payload), encoding="utf-8")

    mutations = module.collect(tmp_path)

    assert mutations == [
        {
            "task_id": "task-1",
            "episode_id": "episode-1",
            "step": 0,
            "tool_name": "slack_conversations_add_message",
            "arguments": {"channel_id": "C1", "text": "hello"},
        }
    ]


def test_mcp_atlas_mutation_report_compares_state_digests(tmp_path):
    module = _load_mutation_report()
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps({"services": {"slack": {"sha256": "old"}, "notion": {"sha256": "same"}}}))
    after.write_text(json.dumps({"services": {"slack": {"sha256": "new"}, "notion": {"sha256": "same"}}}))

    assert module.changed_services(before, after) == ["slack"]


def test_private_report_does_not_change_existing_parent_permissions(tmp_path):
    module = _load_mutation_report()
    tmp_path.chmod(0o755)
    output = tmp_path / "state.json"

    module.write_private(output, {"format": 1})

    assert output.stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 == 0o755


def test_formal_eval_forces_read_only_clean_state_and_cache_reset():
    script = (Path(__file__).parents[1] / "experiments/evals.sh").read_text(encoding="utf-8")

    assert "MCP_ATLAS_READ_ONLY=true" in script
    assert "refuses --mcp-atlas-skip-state-check true" in script
    assert 'f"{sys.argv[1].rstrip(\'/\')}/cache-clear"' in script
