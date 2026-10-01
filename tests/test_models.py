import numpy as np
import pytest
import torch

from egoagent.classical import classical_baselines, cluster_quality
from egoagent.models import ActivityModel, VideoEncoder
from egoagent.training import TrainConfig, evaluate, predict, train_ddp, train_model


@pytest.mark.parametrize("modality", ["video", "imu", "fusion"])
def test_model_shapes(modality):
    model = ActivityModel(5, modality, dim=32)
    video, imu = torch.rand(3, 8, 32, 32), torch.randn(3, 100, 6)
    assert model(video, imu).shape == (3, 5)
    assert model.embed(video, imu).shape == (3, 32)


def test_motion_channels_remove_static_background():
    static = torch.rand(1, 1, 32, 32).repeat(2, 8, 1, 1)
    channels = VideoEncoder.motion_channels(static)
    assert channels.shape == (2, 8, 2, 32, 32)
    assert channels.abs().max() < 1e-6  # nothing moves, so both channels are empty


def test_training_lowers_loss_and_beats_chance(split):
    train, test = split
    model, history = train_model(train, TrainConfig(modality="imu", epochs=6, batch_size=32))
    assert history[-1] < history[0]
    assert evaluate(model, test)["accuracy"] > 0.4  # chance is 0.2


def test_imu_normalisation_is_stored_in_the_model(split):
    train, _ = split
    model, _ = train_model(train, TrainConfig(modality="imu", epochs=1))
    expected = torch.from_numpy(train.imu.reshape(-1, 6)).mean(0)
    torch.testing.assert_close(model.imu_mean, expected)
    assert "imu_mean" in model.state_dict()


def test_predict_returns_probabilities(split):
    train, test = split
    model, _ = train_model(train, TrainConfig(modality="imu", epochs=1))
    probs, emb = predict(model, test)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert emb.shape == (len(test), 64)


def test_ddp_ranks_end_with_identical_weights(split):
    train, test = split
    model, info = train_ddp(train, TrainConfig(modality="imu", epochs=2, batch_size=32), world_size=2)
    sums = info["rank_checksums"]
    assert len(sums) == 2 and sums[0] == pytest.approx(sums[1], rel=1e-6)
    assert len(info["history"]) == 2
    assert evaluate(model, test)["accuracy"] >= 0.0  # loads and runs


def test_classical_baselines_beat_chance(split):
    train, test = split
    results = classical_baselines(train, test)
    assert set(results) == {"gradient_boosting", "svm_rbf"}
    assert all(r["accuracy"] > 0.5 for r in results.values())


def test_cluster_quality_is_perfect_on_separated_blobs():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 20)
    points = np.eye(3)[labels] * 10 + rng.normal(0, 0.1, (60, 3))
    scores = cluster_quality(points, labels, k=3)
    assert scores["adjusted_rand_index"] == pytest.approx(1.0)
    assert scores["nmi"] == pytest.approx(1.0)
