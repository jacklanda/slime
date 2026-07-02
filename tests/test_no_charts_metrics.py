from pathlib import Path


def test_slime_source_does_not_log_charts_metrics():
    slime_dir = Path(__file__).resolve().parents[1] / "slime"

    offenders = []
    for path in slime_dir.rglob("*.py"):
        if "Charts/" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(slime_dir.parent).as_posix())

    assert offenders == []
