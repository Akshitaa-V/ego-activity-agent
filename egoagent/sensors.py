"""Multisensor alignment: estimate the IMU clock offset and cut aligned windows.

The camera and the IMU sample at different rates (10 Hz and 50 Hz) and their clocks need
not agree. ``estimate_offset`` recovers the offset by cross-correlating a motion signal
from each sensor; ``make_windows`` then cuts fixed-length windows on the camera clock and
pulls the matching IMU samples after correcting for that offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synthetic import ACTIVITIES, IMU_HZ, Session


def video_motion_energy(frames: np.ndarray) -> np.ndarray:
    """Mean absolute frame difference per frame (first value repeats the second)."""
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return np.concatenate([diff[:1], diff])


def imu_motion_energy(imu: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Magnitude of the de-meaned IMU signal, lightly smoothed."""
    centred = imu - imu.mean(axis=0, keepdims=True)
    mag = np.linalg.norm(centred, axis=1)
    kernel = np.ones(smooth) / smooth
    return np.convolve(mag, kernel, mode="same")


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, n)
    return np.convolve(x, np.ones(n) / n, mode="same")


def estimate_offset(session: Session, max_lag_s: float = 1.0, step_s: float = 0.02, smooth_s: float = 0.5) -> float:
    """Estimate how far the IMU clock is ahead of the camera clock, in seconds.

    Raw motion energy does not rank activities the same way in both sensors (typing is
    busy for the IMU but quiet for the camera), so correlating it directly is unreliable.
    Activity *changes* show up in both, though. Each energy signal is smoothed and turned
    into a change signal (absolute gradient); for each candidate lag the IMU change signal
    is resampled onto the camera clock and correlated with the video one.
    """
    fps = 1.0 / float(np.median(np.diff(session.frame_times)))
    imu_hz = 1.0 / float(np.median(np.diff(session.imu_times)))
    v = video_motion_energy(session.frames)
    v_change = _zscore(np.abs(np.gradient(_smooth(v, int(smooth_s * fps)))))
    e = _smooth(imu_motion_energy(session.imu), int(smooth_s * imu_hz))
    lags = np.arange(-max_lag_s, max_lag_s + 1e-9, step_s)
    scores = np.empty(lags.size)
    for i, lag in enumerate(lags):
        sampled = np.interp(session.frame_times + lag, session.imu_times, e)
        scores[i] = float(np.dot(v_change, _zscore(np.abs(np.gradient(sampled))))) / v.size
    return float(lags[int(np.argmax(scores))])


@dataclass
class WindowSet:
    """Aligned, fixed-size windows ready for training."""

    video: np.ndarray  # (N, T, H, W) float32
    imu: np.ndarray  # (N, S, 6) float32
    labels: np.ndarray  # (N,) int64 index into ACTIVITIES
    session_ids: np.ndarray  # (N,) int64
    t_start: np.ndarray  # (N,) float64, camera clock
    t_end: np.ndarray  # (N,) float64

    def __len__(self) -> int:
        return int(self.labels.size)

    def subset(self, mask: np.ndarray) -> WindowSet:
        return WindowSet(
            self.video[mask],
            self.imu[mask],
            self.labels[mask],
            self.session_ids[mask],
            self.t_start[mask],
            self.t_end[mask],
        )


def _majority_label(session: Session, t0: float, t1: float) -> int:
    overlap = dict.fromkeys(ACTIVITIES, 0.0)
    for seg in session.segments:
        overlap[seg.activity] += max(0.0, min(seg.end, t1) - max(seg.start, t0))
    return ACTIVITIES.index(max(overlap, key=lambda k: overlap[k]))


def make_windows(
    sessions: list[Session],
    window_s: float = 2.0,
    hop_s: float = 1.0,
    n_frames: int = 8,
    offsets: dict[int, float] | None = None,
) -> WindowSet:
    """Cut windows on the camera clock. ``offsets`` maps session id to the IMU offset to
    correct for; sessions not in the map are treated as already synchronised."""
    offsets = offsets or {}
    n_imu = round(window_s * IMU_HZ)
    video, imu, labels, sids, starts, ends = [], [], [], [], [], []
    for s in sessions:
        imu_on_camera_clock = s.imu_times - offsets.get(s.session_id, 0.0)
        for t0 in np.arange(0.0, s.duration - window_s + 1e-9, hop_s):
            t1 = t0 + window_s
            f_idx = np.flatnonzero((s.frame_times >= t0) & (s.frame_times < t1))
            pick = f_idx[np.linspace(0, f_idx.size - 1, n_frames).round().astype(int)]
            # nearest IMU sample to t0 (robust to float error in the corrected timestamps)
            start = int(np.searchsorted(imu_on_camera_clock, t0 - 0.5 / IMU_HZ))
            idx = np.clip(np.arange(start, start + n_imu), 0, s.imu.shape[0] - 1)
            video.append(s.frames[pick])
            imu.append(s.imu[idx])
            labels.append(_majority_label(s, t0, t1))
            sids.append(s.session_id)
            starts.append(t0)
            ends.append(t1)
    return WindowSet(
        np.stack(video).astype(np.float32),
        np.stack(imu).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sids, dtype=np.int64),
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends, dtype=np.float64),
    )
