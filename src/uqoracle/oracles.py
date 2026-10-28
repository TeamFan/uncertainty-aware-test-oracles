"""Learned test oracles: ensembles that predict the verdict of a test case.

A learned oracle maps a test-case description to P(fail). Two model families
are used so that both within-family (ensemble) and cross-family disagreement
can be measured:

* ``MLPEnsemble``: M multilayer perceptrons, each trained on a bootstrap
  resample with its own seed (a deep ensemble);
* ``GBEnsemble``: M histogram gradient-boosting classifiers on bootstrap
  resamples.

A quantile regressor for a non-functional property (peak deceleration) is
provided for the conformalised-interval experiment.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .simulator import IDX


def features(tests: np.ndarray) -> np.ndarray:
    """Oracle input: raw parameters plus two physically motivated features.

    The extra features are the initial time headway and the kinematic stopping
    margin of the lead-braking manoeuvre; they encode domain knowledge a test
    engineer would supply, not information about the outcome.
    """
    x = np.asarray(tests, dtype=float)
    v_e = x[:, IDX["v_ego"]]
    v_l = x[:, IDX["v_lead"]]
    gap = x[:, IDX["gap"]]
    mu = x[:, IDX["mu"]]
    headway = gap / np.maximum(v_e, 1.0)
    lead_stop = (v_l ** 2 - (x[:, IDX["ratio"]] * v_l) ** 2) / (2.0 * np.minimum(x[:, IDX["decel"]], mu * 9.81))
    ego_stop = v_e ** 2 / (2.0 * mu * 9.81)
    margin = gap + lead_stop - ego_stop
    return np.column_stack([x, headway, margin / 10.0])


class _Ensemble:
    def __init__(self, members: int = 5, seed: int = 0):
        self.members = members
        self.seed = seed
        self.models: list = []
        self.scaler = StandardScaler()

    def _make(self, seed: int):  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, tests: np.ndarray, fail: np.ndarray) -> "_Ensemble":
        z = self.scaler.fit_transform(features(tests))
        y = np.asarray(fail, dtype=int)
        rng = np.random.default_rng(self.seed)
        self.models = []
        for m in range(self.members):
            idx = rng.integers(0, len(z), len(z))
            model = self._make(self.seed * 100 + m)
            model.fit(z[idx], y[idx])
            self.models.append(model)
        return self

    def member_proba(self, tests: np.ndarray) -> np.ndarray:
        """Array of shape (members, n) with P(fail) from each member."""
        z = self.scaler.transform(features(tests))
        return np.stack([m.predict_proba(z)[:, 1] for m in self.models])

    def proba(self, tests: np.ndarray) -> np.ndarray:
        return self.member_proba(tests).mean(axis=0)


class MLPEnsemble(_Ensemble):
    def _make(self, seed: int):
        return MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3, learning_rate_init=2e-3,
                             max_iter=400, early_stopping=True, validation_fraction=0.15,
                             n_iter_no_change=20, random_state=seed)


class GBEnsemble(_Ensemble):
    def _make(self, seed: int):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=seed)


class Novelty:
    """Mean distance to the k nearest training test cases (standardised features)."""

    def __init__(self, k: int = 10):
        self.k = k
        self.scaler = StandardScaler()

    def fit(self, tests: np.ndarray) -> "Novelty":
        z = self.scaler.fit_transform(features(tests))
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(z)
        return self

    def score(self, tests: np.ndarray) -> np.ndarray:
        d, _ = self.nn.kneighbors(self.scaler.transform(features(tests)))
        return d.mean(axis=1)


class QuantileRegressor:
    """Lower/upper quantile models for a continuous non-functional output."""

    def __init__(self, alpha: float = 0.1, seed: int = 0):
        self.alpha = alpha
        self.lo = HistGradientBoostingRegressor(loss="quantile", quantile=alpha / 2, max_iter=300,
                                                learning_rate=0.06, random_state=seed)
        self.hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alpha / 2, max_iter=300,
                                                learning_rate=0.06, random_state=seed)

    def fit(self, tests: np.ndarray, y: np.ndarray) -> "QuantileRegressor":
        z = features(tests)
        self.lo.fit(z, y)
        self.hi.fit(z, y)
        return self

    def predict(self, tests: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = features(tests)
        return self.lo.predict(z), self.hi.predict(z)
