import numpy as np
import pytest

from egoagent.classical import imu_features, video_features
from egoagent.sensors import estimate_offset, make_windows
from egoagent.synthetic import ACTIVITIES, IMU_HZ, VIDEO_FPS, make_session


def test_session_streams_have_expected_rates_and_ranges():
    s = make_session(0, duration=20.0, seed=0)
    assert s.frames.shape == (20 * VIDEO_FPS, 32, 32)
    assert s.imu.shape == (20 * IMU_HZ, 6)
    assert s.frames.min() >= 0.0 and s.frames.max() <= 1.0
    assert s.segments[0].start == 0.0 and s.segments[-1].end == pytest.approx(20.0)
    for a, b in zip(s.segments, s.segments[1:], strict=False):
        assert a.end == b.start and a.activity != b.activity


def test_generation_is_reproducible():
    a, b = make_session(2, 15.0, seed=5), make_session(2, 15.0, seed=5)
    np.testing.assert_array_equal(a.frames, b.frames)
    np.testing.assert_array_equal(a.imu, b.imu)


def test_imu_clock_offset_is_applied_to_timestamps(offset_session):
    assert offset_session.imu_times[0] == pytest.approx(0.4)


def test_offset_estimate_recovers_injected_offset(offset_session):
    assert estimate_offset(offset_session) == pytest.approx(0.4, abs=0.1)


@pytest.mark.parametrize("offset", [-0.6, 0.0, 0.25])
def test_offset_estimate_handles_both_directions(offset):
    s = make_session(11, duration=60.0, seed=3, imu_offset_s=offset)
    assert estimate_offset(s) == pytest.approx(offset, abs=0.1)


def test_windows_shapes_and_labels(small_windows, small_sessions):
    ws = small_windows
    assert ws.video.shape[1:] == (8, 32, 32)
    assert ws.imu.shape[1:] == (100, 6)
    assert len(ws) == ws.labels.size == ws.t_start.size
    s0 = small_sessions[0]
    first = np.flatnonzero(ws.session_ids == 0)[0]
    mid = (ws.t_start[first] + ws.t_end[first]) / 2
    # a 2 s window inside a >=3 s segment is labelled with that segment's activity
    assert ACTIVITIES[ws.labels[first]] in {s0.activity_at(ws.t_start[first]), s0.activity_at(mid)}


def test_offset_correction_realigns_imu(offset_session):
    aligned = make_windows([offset_session], offsets={3: 0.4})
    reference = make_session(3, duration=60.0, seed=7, imu_offset_s=0.0)
    expected = make_windows([reference])
    np.testing.assert_allclose(aligned.imu, expected.imu, atol=1e-5)


def test_imu_features_find_walking_step_frequency():
    s = make_session(0, duration=60.0, seed=0)
    walking = next(seg for seg in s.segments if seg.activity == "walking" and seg.end - seg.start >= 3)
    i0 = int(walking.start * IMU_HZ)
    window = s.imu[i0 : i0 + 2 * IMU_HZ][None]
    feats = imu_features(window)
    dominant_z = feats[0, 4 * 6 + 2]  # block 4 = dominant frequency, channel 2 = acc z
    assert dominant_z == pytest.approx(2.0, abs=0.5)


def test_video_features_shape(small_windows):
    assert video_features(small_windows.video).shape == (len(small_windows), 5)
