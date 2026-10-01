import numpy as np
import pytest

from egoagent.sensors import WindowSet, make_windows
from egoagent.synthetic import make_session, make_sessions


@pytest.fixture(scope="session")
def small_sessions():
    return make_sessions(6, duration=30.0, seed=1)


@pytest.fixture(scope="session")
def small_windows(small_sessions) -> WindowSet:
    return make_windows(small_sessions)


@pytest.fixture(scope="session")
def split(small_windows):
    train = small_windows.subset(small_windows.session_ids < 4)
    test = small_windows.subset(small_windows.session_ids >= 4)
    return train, test


@pytest.fixture()
def offset_session():
    return make_session(3, duration=60.0, seed=7, imu_offset_s=0.4)


@pytest.fixture()
def toy_index():
    """Hand-built index: session 0 is idle 0-3s then stirring 3-6s; session 1 is stirring."""
    from egoagent.index import MomentIndex

    sids = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    starts = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=float)
    ends = starts + 2
    probs = np.zeros((8, 5))
    pred = [0, 0, 0, 2, 2, 2, 2, 2]  # idle=0, stirring=2
    probs[np.arange(8), pred] = 0.9
    probs[np.arange(8), 1] += 0.1
    emb = np.eye(8, 4)[[0, 0, 0, 1, 1, 1, 1, 2]] + 0.01
    return MomentIndex(sids, starts, ends, probs, emb)
