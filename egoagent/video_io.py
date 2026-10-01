"""Load real video into the same layout as the synthetic sessions (needs ``opencv-python``)."""

from __future__ import annotations

import numpy as np


def load_video(path: str, target_fps: float = 10.0, size: int = 32, rgb: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Read a video file, resample to ``target_fps`` and resize to ``size`` x ``size``.

    Returns ``(frame_times, frames)``. Frames are float32 in [0, 1], grey ``(F, H, W)`` by
    default or uint8 ``(F, H, W, 3)`` RGB when ``rgb=True`` (the format CLIP expects).
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, round(src_fps / target_fps))
    frames, times, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            if rgb:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
            times.append(i / src_fps)
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.asarray(times), np.stack(frames)
