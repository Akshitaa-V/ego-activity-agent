"""Synthetic egocentric sessions: a first-person video stream plus a 6-axis IMU stream.

Real egocentric datasets need licences and large downloads, so this module generates
small sessions whose ground truth is known exactly. Each session is a sequence of
activity segments. The camera runs at 10 fps, the IMU at 50 Hz, and the IMU clock can
be shifted against the camera clock to mimic an unsynchronised sensor rig.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIVITIES: tuple[str, ...] = ("idle", "walking", "stirring", "typing", "reaching")
VIDEO_FPS = 10
IMU_HZ = 50
FRAME_SIZE = 32
GRAVITY = 9.81


@dataclass(frozen=True)
class Segment:
    """One stretch of a single activity, in seconds on the camera clock."""

    activity: str
    start: float
    end: float


@dataclass
class Session:
    """A recorded session with two sensor streams on their own clocks."""

    session_id: int
    duration: float
    segments: list[Segment]
    frame_times: np.ndarray  # (F,) seconds, camera clock
    frames: np.ndarray  # (F, H, W) float32 in [0, 1]
    imu_times: np.ndarray  # (N,) seconds, IMU clock (may be offset)
    imu: np.ndarray  # (N, 6) float32: acc xyz (m/s^2), gyro xyz (rad/s)
    imu_offset_s: float = 0.0

    def activity_at(self, t: float) -> str:
        """Ground-truth activity at camera time ``t``."""
        for seg in self.segments:
            if seg.start <= t < seg.end:
                return seg.activity
        return self.segments[-1].activity


def _make_segments(duration: float, rng: np.random.Generator) -> list[Segment]:
    segments: list[Segment] = []
    t = 0.0
    previous = ""
    while t < duration:
        choices = [a for a in ACTIVITIES if a != previous]
        activity = str(rng.choice(choices))
        length = float(rng.uniform(3.0, 8.0))
        end = min(t + length, duration)
        segments.append(Segment(activity, t, end))
        previous = activity
        t = end
    return segments


def _blob(xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2))


def _render_frame(
    activity: str,
    tau: float,
    background: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    noise: float,
) -> np.ndarray:
    """Draw one first-person frame. ``tau`` is the time since the segment started."""
    xx, yy = grid
    if activity == "walking":
        shift = int(tau * 12) % FRAME_SIZE  # ego-motion: the scene scrolls past
        frame = np.roll(background, shift, axis=0) + 0.25 * np.sin(2 * np.pi * (yy * 4 + tau * 1.5))
    else:
        frame = background.copy()
    if activity == "stirring":  # hand circling over a pot
        angle = 2 * np.pi * 1.5 * tau
        frame = frame + _blob(xx, yy, 0.5 + 0.15 * np.cos(angle), 0.7 + 0.1 * np.sin(angle), 0.08)
    elif activity == "typing":  # two hands near the bottom edge, small jitter
        for cx in (0.35, 0.65):
            frame = frame + 0.8 * _blob(xx, yy, cx, 0.85 + rng.normal(0, 0.02), 0.06)
    elif activity == "reaching":  # hand travels up into the scene
        progress = (tau % 2.0) / 2.0
        frame = frame + _blob(xx, yy, 0.5, 0.95 - 0.65 * progress, 0.09)
    frame = frame + rng.normal(0, noise, frame.shape)
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def _imu_signal(activity: str, tau: np.ndarray, amp: float) -> np.ndarray:
    """Deterministic part of the IMU signal for one segment."""
    out = np.zeros((tau.size, 6), dtype=np.float64)
    out[:, 2] = GRAVITY
    two_pi = 2 * np.pi
    if activity == "walking":
        out[:, 2] += amp * 2.0 * np.sin(two_pi * 2.0 * tau)
        out[:, 0] += amp * 0.8 * np.sin(two_pi * 1.0 * tau)
        out[:, 4] += amp * 0.3 * np.sin(two_pi * 1.0 * tau)
    elif activity == "stirring":
        out[:, 0] += amp * 0.6 * np.cos(two_pi * 1.5 * tau)
        out[:, 1] += amp * 0.6 * np.sin(two_pi * 1.5 * tau)
        out[:, 5] += amp * 1.2 * np.sin(two_pi * 1.5 * tau)
    elif activity == "typing":
        out[:, 2] += amp * 0.25 * np.sin(two_pi * 8.0 * tau)
        out[:, 3] += amp * 0.1 * np.sin(two_pi * 8.0 * tau)
    elif activity == "reaching":
        phase = (tau % 2.0) / 2.0
        out[:, 1] += amp * 1.5 * np.sin(np.pi * phase)
        out[:, 3] += amp * 0.4 * np.cos(np.pi * phase)
    return out


def make_session(
    session_id: int,
    duration: float = 60.0,
    seed: int = 0,
    imu_offset_s: float = 0.0,
    video_noise: float = 0.08,
    imu_noise: float = 0.15,
) -> Session:
    """Generate one session. ``imu_offset_s`` > 0 means the IMU clock runs ahead."""
    rng = np.random.default_rng([seed, session_id])
    segments = _make_segments(duration, rng)

    lin = np.linspace(0.0, 1.0, FRAME_SIZE)
    xx, yy = np.meshgrid(lin, lin)
    background = 0.2 + 0.3 * rng.random() * (np.sin(xx * rng.uniform(2, 6)) * np.cos(yy * rng.uniform(2, 6)) + 1)

    frame_times = np.arange(0.0, duration, 1.0 / VIDEO_FPS)
    frames = np.empty((frame_times.size, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    true_imu_times = np.arange(0.0, duration, 1.0 / IMU_HZ)
    imu = np.empty((true_imu_times.size, 6), dtype=np.float64)

    for seg in segments:
        amp = float(rng.uniform(0.7, 1.3))
        f_mask = (frame_times >= seg.start) & (frame_times < seg.end)
        for i in np.flatnonzero(f_mask):
            frames[i] = _render_frame(seg.activity, frame_times[i] - seg.start, background, (xx, yy), rng, video_noise)
        i_mask = (true_imu_times >= seg.start) & (true_imu_times < seg.end)
        imu[i_mask] = _imu_signal(seg.activity, true_imu_times[i_mask] - seg.start, amp)

    imu[:, :3] += rng.normal(0, imu_noise, (imu.shape[0], 3))
    imu[:, 3:] += rng.normal(0, imu_noise / 3, (imu.shape[0], 3))

    return Session(
        session_id=session_id,
        duration=duration,
        segments=segments,
        frame_times=frame_times,
        frames=frames,
        imu_times=true_imu_times + imu_offset_s,
        imu=imu.astype(np.float32),
        imu_offset_s=imu_offset_s,
    )


def make_sessions(n: int, duration: float = 60.0, seed: int = 0, max_offset_s: float = 0.0) -> list[Session]:
    """Generate ``n`` sessions, each with a random IMU clock offset in ``[-max_offset_s, max_offset_s]``."""
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-max_offset_s, max_offset_s, n) if max_offset_s > 0 else np.zeros(n)
    return [make_session(i, duration, seed, float(offsets[i])) for i in range(n)]
