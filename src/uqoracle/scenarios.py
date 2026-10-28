"""Test-case generators for the operational design domain (ODD) and its shifts."""
from __future__ import annotations

import numpy as np

from .simulator import IDX, PARAMS

# Nominal ODD used to train and calibrate the learned oracle.
NOMINAL = {
    "v_ego": (10.0, 35.0),
    "v_lead_frac": (0.3, 1.05),
    "gap_brake": (8.0, 70.0),
    "gap_cutin": (6.0, 40.0),
    "t_event": (0.5, 5.0),
    "decel": (1.0, 9.0),
    "ratio": (0.0, 0.9),
    "mu": (0.6, 1.0),
    "noise": (0.0, 2.0),
    "latency": (0.0, 0.3),
    "dropout": (0.0, 0.4),
}

# Shifted ODD: wet/snowy roads and slower perception.
SHIFTED = dict(NOMINAL, mu=(0.3, 0.6), latency=(0.2, 0.5))


def sample_tests(n: int, rng: np.random.Generator, odd: dict | None = None,
                 cut_in_share: float = 0.4) -> np.ndarray:
    """Draw ``n`` test cases uniformly from an ODD description."""
    odd = NOMINAL if odd is None else odd
    u = lambda key: rng.uniform(*odd[key], size=n)  # noqa: E731
    x = np.zeros((n, len(PARAMS)))
    cut = rng.random(n) < cut_in_share
    x[:, IDX["cut_in"]] = cut
    x[:, IDX["v_ego"]] = u("v_ego")
    x[:, IDX["v_lead"]] = x[:, IDX["v_ego"]] * u("v_lead_frac")
    x[:, IDX["gap"]] = np.where(cut, u("gap_cutin"), u("gap_brake"))
    x[:, IDX["t_event"]] = u("t_event")
    x[:, IDX["decel"]] = u("decel")
    x[:, IDX["ratio"]] = u("ratio")
    x[:, IDX["mu"]] = u("mu")
    x[:, IDX["noise"]] = u("noise")
    x[:, IDX["latency"]] = u("latency")
    x[:, IDX["dropout"]] = u("dropout")
    return x
