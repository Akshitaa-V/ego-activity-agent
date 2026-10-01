"""End-to-end experiment: synthesise data, align sensors, train and compare models,
cluster embeddings, build the moment index and run the agent on a few questions."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import Agent, build_tools
from .classical import classical_baselines, cluster_quality, window_features
from .index import MomentIndex
from .llm import RuleBasedLLM
from .sensors import WindowSet, estimate_offset, make_windows
from .synthetic import ACTIVITIES, make_sessions
from .training import TrainConfig, evaluate, predict, train_ddp, train_model


@dataclass
class PipelineConfig:
    n_sessions: int = 32
    duration_s: float = 60.0
    test_fraction: float = 0.25
    max_offset_s: float = 0.8
    epochs: int = 8
    seed: int = 0
    ddp_world_size: int = 2  # 0 skips the distributed run
    out_dir: str = "artifacts"


def _split(ws: WindowSet, n_sessions: int, test_fraction: float) -> tuple[WindowSet, WindowSet]:
    """Split by session, never by window, so no recording leaks into both sides."""
    n_test = max(1, round(n_sessions * test_fraction))
    test_mask = ws.session_ids >= n_sessions - n_test
    return ws.subset(~test_mask), ws.subset(test_mask)


def _timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, round(time.perf_counter() - start, 1)


def run_pipeline(cfg: PipelineConfig, log: Any = print) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"config": asdict(cfg), "activities": list(ACTIVITIES)}

    log(f"Generating {cfg.n_sessions} sessions of {cfg.duration_s:.0f}s (camera 10 Hz, IMU 50 Hz)")
    sessions = make_sessions(cfg.n_sessions, cfg.duration_s, cfg.seed, cfg.max_offset_s)

    estimated = {s.session_id: estimate_offset(s, max_lag_s=cfg.max_offset_s + 0.2) for s in sessions}
    errors = np.array([estimated[s.session_id] - s.imu_offset_s for s in sessions])
    results["clock_sync"] = {
        "true_offset_range_s": [
            round(float(min(s.imu_offset_s for s in sessions)), 3),
            round(float(max(s.imu_offset_s for s in sessions)), 3),
        ],
        "mean_abs_error_ms": round(float(np.abs(errors).mean() * 1000), 1),
        "max_abs_error_ms": round(float(np.abs(errors).max() * 1000), 1),
    }
    log(f"Clock offsets recovered: {results['clock_sync']}")

    aligned = make_windows(sessions, offsets=estimated)
    unaligned = make_windows(sessions)
    train, test = _split(aligned, cfg.n_sessions, cfg.test_fraction)
    train_raw_clock, test_raw_clock = _split(unaligned, cfg.n_sessions, cfg.test_fraction)
    results["data"] = {
        "train_windows": len(train),
        "test_windows": len(test),
        "class_counts_test": np.bincount(test.labels, minlength=len(ACTIVITIES)).tolist(),
    }

    log("Classical baselines on hand-crafted features (gradient boosting, SVM)")
    results["classical"], _ = _timed(classical_baselines, train, test, cfg.seed)
    log(f"  {results['classical']}")

    runs = {
        "imu_only": TrainConfig(modality="imu"),
        "video_only_raw_frames": TrainConfig(modality="video"),
        "video_only_motion_input": TrainConfig(modality="video", motion_input=True),
        "fusion": TrainConfig(modality="fusion"),
    }
    results["deep"] = {}
    models = {}
    for name, tcfg in runs.items():
        tcfg.epochs, tcfg.seed = cfg.epochs, cfg.seed
        (model, history), secs = _timed(train_model, train, tcfg)
        metrics = evaluate(model, test)
        results["deep"][name] = {
            "accuracy": round(metrics["accuracy"], 3),
            "macro_f1": round(metrics["macro_f1"], 3),
            "final_train_loss": round(history[-1], 4),
            "seconds": secs,
        }
        models[name] = model
        log(f"  {name}: {results['deep'][name]}")
        if name == "fusion":
            results["fusion_confusion"] = metrics["confusion"]

    fusion_cfg = TrainConfig(modality="fusion", epochs=cfg.epochs, seed=cfg.seed)
    (model_nosync, _), secs = _timed(train_model, train_raw_clock, fusion_cfg)
    m = evaluate(model_nosync, test_raw_clock)
    results["deep"]["fusion_without_clock_sync"] = {
        "accuracy": round(m["accuracy"], 3),
        "macro_f1": round(m["macro_f1"], 3),
        "seconds": secs,
    }
    log(f"  fusion_without_clock_sync: {results['deep']['fusion_without_clock_sync']}")

    if cfg.ddp_world_size > 1:
        log(f"Distributed training: DDP over {cfg.ddp_world_size} CPU processes (gloo)")
        (ddp_model, info), _ = _timed(train_ddp, train, fusion_cfg, cfg.ddp_world_size)
        m = evaluate(ddp_model, test)
        sums = info["rank_checksums"]
        results["ddp"] = {
            "world_size": cfg.ddp_world_size,
            "accuracy": round(m["accuracy"], 3),
            "seconds": round(info["seconds"], 1),
            "weights_identical_across_ranks": bool(np.allclose(sums, sums[0], rtol=1e-6)),
        }
        log(f"  {results['ddp']}")

    probs, emb = predict(models["fusion"], test)
    k = len(ACTIVITIES)
    results["clustering"] = {
        "kmeans_on_learned_embeddings": cluster_quality(emb, test.labels, k, cfg.seed),
        "kmeans_on_handcrafted_features": cluster_quality(window_features(test), test.labels, k, cfg.seed),
    }
    log(f"Clustering (no labels used): {results['clustering']}")

    index = MomentIndex.from_windows(test, probs, emb)
    torch.save(models["fusion"].state_dict(), out / "fusion_model.pt")
    with open(out / "index.pkl", "wb") as f:
        pickle.dump(index, f)

    sid = index.sessions()[0]
    agent = Agent(RuleBasedLLM(), build_tools(index))
    questions = [
        f"When was I stirring in session {sid}?",
        f"How long did I spend on each activity in session {sid}?",
        f"Show me moments similar to session {sid} at 10s",
    ]
    results["agent_demo"] = []
    for q in questions:
        r = agent.run(q)
        results["agent_demo"].append({"question": q, "tools": [t.tool for t in r.trace], "answer": r.answer})
        log(f"Q: {q}\nA: {r.answer}")

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved results, model and index to {out}/")
    return results
