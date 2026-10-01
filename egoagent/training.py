"""Training, evaluation and embedding extraction, on one process or with DDP.

``train_ddp`` runs ``torch.nn.parallel.DistributedDataParallel`` over several CPU
processes with the gloo backend. Each process sees its own shard of the data through a
``DistributedSampler``; gradients are averaged across processes after every backward pass.
The same code runs on GPUs by switching the backend to nccl and moving tensors to devices.
"""

from __future__ import annotations

import os
import socket
import tempfile
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .models import ActivityModel, Modality
from .sensors import WindowSet
from .synthetic import ACTIVITIES


@dataclass
class TrainConfig:
    modality: Modality = "fusion"
    epochs: int = 8
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    dim: int = 64
    motion_input: bool = False  # True = background-removed + frame-difference input (ablation)


def _dataset(ws: WindowSet) -> TensorDataset:
    return TensorDataset(torch.from_numpy(ws.video), torch.from_numpy(ws.imu), torch.from_numpy(ws.labels))


def build_model(train: WindowSet, cfg: TrainConfig) -> ActivityModel:
    torch.manual_seed(cfg.seed)
    model = ActivityModel(
        len(ACTIVITIES), cfg.modality, cfg.dim, n_frames=train.video.shape[1], motion_input=cfg.motion_input
    )
    flat = torch.from_numpy(train.imu.reshape(-1, train.imu.shape[-1]))
    model.set_imu_stats(flat.mean(0), flat.std(0))
    return model


def _run_epochs(
    model: nn.Module, loader: DataLoader, cfg: TrainConfig, sampler: DistributedSampler | None = None
) -> list[float]:
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, cfg.epochs))
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # reshuffle shards differently each epoch
        model.train()
        total, count = 0.0, 0
        for video, imu, y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(video, imu), y)
            loss.backward()
            optimiser.step()
            total += loss.item() * y.size(0)
            count += y.size(0)
        scheduler.step()
        history.append(total / max(count, 1))
    return history


def train_model(train: WindowSet, cfg: TrainConfig) -> tuple[ActivityModel, list[float]]:
    """Train on a single process. Returns the model and the mean loss per epoch."""
    model = build_model(train, cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(_dataset(train), batch_size=cfg.batch_size, shuffle=True, generator=generator)
    return model, _run_epochs(model, loader, cfg)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _ddp_worker(rank: int, world_size: int, port: int, train: WindowSet, cfg_dict: dict, out_path: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        cfg = TrainConfig(**cfg_dict)
        model = DistributedDataParallel(build_model(train, cfg))
        sampler = DistributedSampler(_dataset(train), num_replicas=world_size, rank=rank, seed=cfg.seed)
        loader = DataLoader(_dataset(train), batch_size=cfg.batch_size, sampler=sampler)
        history = _run_epochs(model, loader, cfg, sampler)
        # every rank holds identical weights after each step; check that, then save once
        with torch.no_grad():
            checksum = torch.tensor([sum(p.sum().item() for p in model.parameters())])
        gathered = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather(gathered, checksum)
        if rank == 0:
            torch.save(
                {
                    "state_dict": model.module.state_dict(),
                    "history": history,
                    "checksums": [float(g) for g in gathered],
                },
                out_path,
            )
    finally:
        dist.destroy_process_group()


def train_ddp(train: WindowSet, cfg: TrainConfig, world_size: int = 2) -> tuple[ActivityModel, dict]:
    """Train with DistributedDataParallel over ``world_size`` CPU processes.

    Returns the trained model and run info: per-epoch loss on rank 0, wall time and the
    parameter checksum from every rank (they should match).
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "ddp.pt")
        start = time.perf_counter()
        mp.spawn(
            _ddp_worker,
            args=(world_size, _free_port(), train, asdict(cfg), out_path),
            nprocs=world_size,
            join=True,
        )
        elapsed = time.perf_counter() - start
        saved = torch.load(out_path, weights_only=False)
    model = build_model(train, cfg)
    model.load_state_dict(saved["state_dict"])
    info = {"history": saved["history"], "seconds": elapsed, "rank_checksums": saved["checksums"]}
    return model, info


@torch.no_grad()
def predict(model: ActivityModel, ws: WindowSet, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return class probabilities (N, C) and embeddings (N, dim)."""
    model.eval()
    probs, embs = [], []
    for video, imu, _ in DataLoader(_dataset(ws), batch_size=batch_size):
        emb = model.embed(video, imu)
        probs.append(torch.softmax(model.classify(emb), dim=1).numpy())
        embs.append(emb.numpy())
    return np.concatenate(probs), np.concatenate(embs)


def evaluate(model: ActivityModel, ws: WindowSet) -> dict:
    probs, _ = predict(model, ws)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float((pred == ws.labels).mean()),
        "macro_f1": float(f1_score(ws.labels, pred, average="macro")),
        "confusion": confusion_matrix(ws.labels, pred, labels=range(len(ACTIVITIES))).tolist(),
    }
