import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from egoagent.video_io import load_video  # noqa: E402


def _write_video(path, n_frames=30, fps=30, size=64):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    for i in range(n_frames):
        frame = np.full((size, size, 3), i * 8 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_load_video_resamples_and_resizes(tmp_path):
    path = tmp_path / "clip.avi"
    _write_video(path, n_frames=30, fps=30)
    times, frames = load_video(str(path), target_fps=10, size=32)
    assert frames.shape == (10, 32, 32)  # 1 s at 30 fps -> 10 frames at 10 fps
    assert frames.dtype == np.float32 and frames.min() >= 0.0 and frames.max() <= 1.0
    np.testing.assert_allclose(np.diff(times), 0.1, atol=1e-6)


def test_load_video_rgb_mode_for_clip(tmp_path):
    path = tmp_path / "clip.avi"
    _write_video(path, n_frames=6, fps=30)
    _, frames = load_video(str(path), target_fps=10, size=24, rgb=True)
    assert frames.shape == (2, 24, 24, 3) and frames.dtype == np.uint8


def test_load_video_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_video(str(tmp_path / "missing.mp4"))
