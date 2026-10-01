"""Does correcting the IMU clock offset help fusion? Repeat over seeds on the small setting.

Run: python experiments/clock_sync_seeds.py
"""

from __future__ import annotations

import json

import numpy as np

from egoagent.sensors import estimate_offset, make_windows
from egoagent.synthetic import make_sessions
from egoagent.training import TrainConfig, evaluate, train_model


def main(n_sessions: int = 16, n_test: int = 4, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    rows = []
    for seed in seeds:
        sessions = make_sessions(n_sessions, 60.0, seed, max_offset_s=0.8)
        offsets = {s.session_id: estimate_offset(s, max_lag_s=1.0) for s in sessions}
        row = {"seed": seed}
        for name, ws in (("synced", make_windows(sessions, offsets=offsets)), ("unsynced", make_windows(sessions))):
            test_mask = ws.session_ids >= n_sessions - n_test
            model, _ = train_model(ws.subset(~test_mask), TrainConfig(modality="fusion", epochs=8, seed=seed))
            row[name] = round(evaluate(model, ws.subset(test_mask))["accuracy"], 3)
        rows.append(row)
        print(row)
    summary = {
        "runs": rows,
        "mean_synced": round(float(np.mean([r["synced"] for r in rows])), 3),
        "mean_unsynced": round(float(np.mean([r["unsynced"] for r in rows])), 3),
    }
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
