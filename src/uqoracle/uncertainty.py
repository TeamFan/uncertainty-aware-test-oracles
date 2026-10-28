"""Uncertainty scores for binary oracle verdicts and their evaluation metrics.

Convention: every score is oriented so that *larger means less trustworthy*.

Single-inference scores use one model and one forward pass. For a binary
probability p they are all strictly decreasing functions of |p - 1/2| and
therefore induce the same ranking of test cases (see ``tests/``):

    1 - max(p, 1-p),  binary entropy h(p),  Gini 1 - p^2 - (1-p)^2,  -|2p - 1|.

Multi-inference scores combine several predictions (ensemble members or model
families) and can separate data noise from model disagreement:

    H[mean p] = mean H[p_m] + MI,

where the expected entropy ``mean H[p_m]`` is the aleatoric part and the mutual
information ``MI`` the epistemic part.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

EPS = 1e-12


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def single_pass_scores(p: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "one_minus_msp": 1.0 - np.maximum(p, 1 - p),
        "entropy": binary_entropy(p),
        "gini": 1.0 - p ** 2 - (1 - p) ** 2,
        "neg_margin": -np.abs(2 * p - 1),
    }


def ensemble_scores(member_p: np.ndarray) -> dict[str, np.ndarray]:
    """``member_p`` has shape (members, n)."""
    mean = member_p.mean(axis=0)
    total = binary_entropy(mean)
    aleatoric = binary_entropy(member_p).mean(axis=0)
    return {
        "total_entropy": total,
        "aleatoric": aleatoric,
        "mutual_info": np.maximum(total - aleatoric, 0.0),
        "member_std": member_p.std(axis=0),
    }


# --------------------------------------------------------------------------- metrics

def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (ties = 1/2)."""
    score = np.asarray(score, float)
    pos = np.asarray(positive, bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)  # average ranks for ties
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def risk_coverage(score: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accept the most certain verdicts first; return (coverage, selective risk)."""
    order = np.argsort(score, kind="mergesort")
    err = np.asarray(error, float)[order]
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def aurc(score: np.ndarray, error: np.ndarray) -> float:
    _, risk = risk_coverage(score, error)
    return float(risk.mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of P(fail) with equal-width bins."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)
