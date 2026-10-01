import json

import pytest

from egoagent.cli import main


@pytest.mark.slow
def test_quick_demo_then_ask(tmp_path, capsys):
    assert main(["demo", "--quick", "--ddp", "0", "--out", str(tmp_path)]) == 0
    results = json.loads((tmp_path / "results.json").read_text())
    for key in ("clock_sync", "classical", "deep", "clustering", "agent_demo"):
        assert key in results
    assert (tmp_path / "index.pkl").exists() and (tmp_path / "fusion_model.pt").exists()

    capsys.readouterr()
    assert main(["ask", "How long did I spend on each activity in session 7?", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "[tool] activity_stats" in out and "Time per activity" in out


def test_ask_without_index_fails_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        main(["ask", "anything", "--out", str(tmp_path)])
