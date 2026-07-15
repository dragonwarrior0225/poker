import numpy as np

from neurons.detector import HumanAnomalyDetector


def test_human_anomaly_detector_ranks_shifted_rows_higher():
    rng = np.random.default_rng(44)
    human = rng.normal(0.0, 1.0, size=(200, 8))
    bot = rng.normal(5.0, 1.0, size=(40, 8))
    X = np.vstack([human, bot])
    y = np.concatenate([np.zeros(len(human)), np.ones(len(bot))])

    detector = HumanAnomalyDetector(n_estimators=50).fit(X, y)
    probabilities = detector.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[len(human) :, 1].mean() > probabilities[: len(human), 1].mean()
