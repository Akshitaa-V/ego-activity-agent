"""PyTorch encoders: a CNN + Transformer for video clips, a 1D CNN for IMU windows,
and a late-fusion head. ``modality`` switches between video-only, IMU-only and fusion so
the same code runs the sensor ablation."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

Modality = Literal["video", "imu", "fusion"]


class FrameEncoder(nn.Module):
    """Small CNN mapping one (C, H, W) frame to a ``dim`` vector."""

    def __init__(self, dim: int = 64, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoEncoder(nn.Module):
    """Per-frame CNN features, then a Transformer encoder over time, mean-pooled.

    By default the CNN sees raw frames. ``motion_input=True`` instead feeds two channels,
    the frame minus the clip's mean frame (static background removed) and the difference
    to the previous frame. That was a hypothesis against per-session background
    overfitting; the ablation in the README shows it did not help, so it is off by default.
    """

    def __init__(
        self, dim: int = 64, n_frames: int = 8, layers: int = 2, heads: int = 4, motion_input: bool = False
    ) -> None:
        super().__init__()
        self.motion_input = motion_input
        self.frame = FrameEncoder(dim, in_channels=2 if motion_input else 1)
        self.pos = nn.Parameter(torch.zeros(1, n_frames, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=0.1, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers)

    @staticmethod
    def motion_channels(video: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W) -> (B, T, 2, H, W): background-removed frame and frame difference."""
        foreground = video - video.mean(dim=1, keepdim=True)
        diff = torch.diff(video, dim=1, prepend=video[:, :1])
        return torch.stack([foreground, diff], dim=2)

    def forward(self, video: torch.Tensor) -> torch.Tensor:  # (B, T, H, W)
        b, t, h, w = video.shape
        if self.motion_input:
            x = self.motion_channels(video).reshape(b * t, 2, h, w)
        else:  # raw frames, kept for the ablation
            x = video.reshape(b * t, 1, h, w)
        feats = self.frame(x).reshape(b, t, -1)
        return self.temporal(feats + self.pos[:, :t]).mean(dim=1)


class IMUEncoder(nn.Module):
    """1D CNN over a (S, 6) IMU window."""

    def __init__(self, dim: int = 64, channels: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, dim, 5, padding=2, stride=2),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:  # (B, S, 6)
        return self.net(imu.transpose(1, 2))


class ActivityModel(nn.Module):
    """Encodes one or both sensor streams into an embedding and classifies the activity.

    IMU normalisation statistics live in buffers so a saved model normalises its inputs
    the same way at inference time.
    """

    def __init__(
        self,
        n_classes: int,
        modality: Modality = "fusion",
        dim: int = 64,
        n_frames: int = 8,
        motion_input: bool = False,
    ) -> None:
        super().__init__()
        self.modality = modality
        use_video = modality in ("video", "fusion")
        self.video_enc = VideoEncoder(dim, n_frames, motion_input=motion_input) if use_video else None
        self.imu_enc = IMUEncoder(dim) if modality in ("imu", "fusion") else None
        in_dim = dim * (2 if modality == "fusion" else 1)
        self.project = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(0.1))
        self.classify = nn.Linear(dim, n_classes)
        self.register_buffer("imu_mean", torch.zeros(6))
        self.register_buffer("imu_std", torch.ones(6))

    def set_imu_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.imu_mean.copy_(mean)
        self.imu_std.copy_(std.clamp_min(1e-6))

    def embed(self, video: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.video_enc is not None:
            parts.append(self.video_enc(video))
        if self.imu_enc is not None:
            parts.append(self.imu_enc((imu - self.imu_mean) / self.imu_std))
        return self.project(torch.cat(parts, dim=1))

    def forward(self, video: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        return self.classify(self.embed(video, imu))
