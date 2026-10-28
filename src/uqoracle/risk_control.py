"""Risk-controlled acceptance of oracle verdicts and budgeted verification.

Acceptance rule: accept the learned oracle's verdict for a test case when its
uncertainty score ``u`` is at most a threshold ``tau``; otherwise execute the
test in simulation. With the per-test loss

    L_i(tau) = 1{u_i <= tau} * l_i,     l_i in [0, 1]  (e.g. l_i = false pass),

L_i is non-decreasing in tau, so conformal risk control (Angelopoulos et al.,
ICLR 2024) gives E[L_test(tau_hat)] <= alpha when calibration and test cases
are exchangeable, with

    tau_hat = max { tau : (n * Rhat_n(tau) + 1) / (n + 1) <= alpha }.
"""
from __future__ import annotations

import numpy as np


def crc_threshold(score: np.ndarray, loss: np.ndarray, alpha: float) -> float:
    """Largest threshold whose inflated empirical risk is at most ``alpha``.

    Returns ``-inf`` (accept nothing) when even the smallest threshold fails.
    """
    score = np.asarray(score, float)
    loss = np.asarray(loss, float)
    n = len(score)
    if (1.0 / (n + 1)) > alpha:
        return -np.inf
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    cum = np.cumsum(loss[order])            # n * Rhat at tau = s[k]
    ok = (cum + 1.0) / (n + 1) <= alpha
    # Thresholds must not split ties: tau = s[k] accepts every score <= s[k].
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    valid = ok & last_of_tie
    # Rhat is non-decreasing, so the feasible set is a prefix; take its end.
    if not valid.any():
        return -np.inf
    k = np.flatnonzero(valid)
    # the prefix property: every k' <= k must also be ok
    first_bad = np.flatnonzero(~ok)
    if first_bad.size:
        k = k[k < first_bad[0]]
        if k.size == 0:
            return -np.inf
    return float(s[k[-1]])


def accept(score: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(score, float) <= tau


def expected_cost(p_fail: np.ndarray, verdict_fail: np.ndarray, c_false_pass: float,
                  c_false_fail: float) -> np.ndarray:
    """Expected cost of accepting each verdict given calibrated P(fail)."""
    p = np.asarray(p_fail, float)
    v = np.asarray(verdict_fail, bool)
    return np.where(v, c_false_fail * (1 - p), c_false_pass * p)


def residual_cost(priority: np.ndarray, verdict_fail: np.ndarray, fail: np.ndarray, budget: int,
                  c_false_pass: float, c_false_fail: float) -> float:
    """Cost of the accepted (unverified) verdicts after executing the ``budget``
    test cases with the highest priority; executed tests are judged correctly."""
    order = np.argsort(-np.asarray(priority, float), kind="mergesort")
    verified = np.zeros(len(order), bool)
    verified[order[:budget]] = True
    v = np.asarray(verdict_fail, bool)
    y = np.asarray(fail, bool)
    fp = (~v) & y & ~verified
    ff = v & (~y) & ~verified
    return float(c_false_pass * fp.sum() + c_false_fail * ff.sum())
