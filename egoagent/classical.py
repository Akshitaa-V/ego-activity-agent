"""Hand-crafted features and classical ML baselines (gradient boosting, SVM, k-means)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .sensors import WindowSet
from .synthetic import IMU_HZ


def imu_features(imu: np.ndarray, hz: int = IMU_HZ) -> np.ndarray:
    """Per-channel statistics for IMU windows of shape (N, S, 6).

    Returns (N, 6 * 6): mean, std, min, max, dominant frequency (Hz) and spectral energy
    of the de-meaned signal for each channel.
    """
    centred = imu - imu.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centred, axis=1)) ** 2
    freqs = np.fft.rfftfreq(imu.shape[1], d=1.0 / hz)
    dominant = freqs[1:][np.argmax(spectrum[:, 1:, :], axis=1)]  # skip the DC bin
    parts = [
        imu.mean(axis=1),
        imu.std(axis=1),
        imu.min(axis=1),
        imu.max(axis=1),
        dominant,
        np.log1p(spectrum.sum(axis=1)),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def video_features(video: np.ndarray) -> np.ndarray:
    """Cheap video statistics for windows of shape (N, T, H, W): brightness, contrast,
    motion energy and where in the frame the motion happens (top vs bottom half)."""
    diff = np.abs(np.diff(video, axis=1))
    half = video.shape[2] // 2
    return np.stack(
        [
            video.mean(axis=(1, 2, 3)),
            video.std(axis=(1, 2, 3)),
            diff.mean(axis=(1, 2, 3)),
            diff[:, :, :half].mean(axis=(1, 2, 3)),
            diff[:, :, half:].mean(axis=(1, 2, 3)),
        ],
        axis=1,
    ).astype(np.float32)


def window_features(ws: WindowSet) -> np.ndarray:
    return np.concatenate([imu_features(ws.imu), video_features(ws.video)], axis=1)


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def classical_baselines(train: WindowSet, test: WindowSet, seed: int = 0) -> dict[str, dict[str, float]]:
    """Fit gradient boosting and an RBF SVM on hand-crafted features."""
    x_tr, x_te = window_features(train), window_features(test)
    models = {
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale")),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_tr, train.labels)
        results[name] = _scores(test.labels, model.predict(x_te))
    return results


def cluster_quality(embeddings: np.ndarray, labels: np.ndarray, k: int, seed: int = 0) -> dict[str, float]:
    """Run k-means without labels, then compare the clusters with the true activities."""
    x = StandardScaler().fit_transform(embeddings)
    clusters = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
    }
