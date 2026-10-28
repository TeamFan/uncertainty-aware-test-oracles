import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uqoracle.risk_control import crc_threshold, expected_cost, residual_cost  # noqa: E402
from uqoracle.scenarios import SHIFTED, sample_tests  # noqa: E402
from uqoracle.simulator import IDX, PARAMS, V1, V2, simulate  # noqa: E402
from uqoracle.uncertainty import (auroc, aurc, binary_entropy, ece, ensemble_scores,  # noqa: E402
                                  single_pass_scores)


# ----------------------------------------------------------------------- uncertainty scores
def test_single_pass_scores_induce_identical_rankings():
    p = np.random.default_rng(0).random(5000)
    scores = single_pass_scores(p)
    ref = np.argsort(np.argsort(scores["one_minus_msp"]))
    for name, s in scores.items():
        assert np.array_equal(np.argsort(np.argsort(s)), ref), name
    err = np.random.default_rng(1).random(5000) < 0.3
    aucs = {name: auroc(s, err) for name, s in scores.items()}
    assert max(aucs.values()) - min(aucs.values()) < 1e-12
    assert len({round(aurc(s, err), 12) for s in scores.values()}) == 1


def test_entropy_decomposition():
    rng = np.random.default_rng(2)
    mp = rng.random((5, 1000))
    s = ensemble_scores(mp)
    assert np.allclose(s["total_entropy"], s["aleatoric"] + s["mutual_info"], atol=1e-12)
    assert (s["mutual_info"] >= 0).all()
    same = np.repeat(rng.random((1, 1000)), 5, axis=0)
    assert np.allclose(ensemble_scores(same)["mutual_info"], 0.0, atol=1e-12)
    assert binary_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2))


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(3)
    score = np.round(rng.random(2000), 2)
    y = rng.random(2000) < 0.2
    assert auroc(score, y) == pytest.approx(roc_auc_score(y, score), abs=1e-12)


def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = np.concatenate([np.arange(1000) < 100, np.arange(1000) < 500, np.arange(1000) < 900])
    assert ece(p, y) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------- risk control
def test_crc_threshold_respects_inflated_empirical_risk():
    rng = np.random.default_rng(4)
    score = rng.random(1000)
    loss = (rng.random(1000) < score * 0.1).astype(float)
    for alpha in [0.005, 0.01, 0.05]:
        tau = crc_threshold(score, loss, alpha)
        r = (loss * (score <= tau)).sum()
        assert (r + 1) / 1001 <= alpha + 1e-12
        # one more accepted case would break the bound (maximality), unless all accepted
        nxt = score[score > tau]
        if nxt.size:
            tau2 = nxt.min()
            r2 = (loss * (score <= tau2)).sum()
            assert (r2 + 1) / 1001 > alpha
    assert crc_threshold(score, loss, 0.0005) == -np.inf


def test_crc_threshold_handles_ties():
    score = np.array([0.1, 0.1, 0.2, 0.3])
    loss = np.array([0.0, 1.0, 0.0, 0.0])
    # alpha = 0.3: (0 + 1)/5 = 0.2 ok below the tie; at the tie (1 + 1)/5 = 0.4 > 0.3
    assert crc_threshold(score, loss, 0.3) == -np.inf or crc_threshold(score, loss, 0.3) < 0.1
    assert crc_threshold(score, loss, 0.45) == 0.3


def test_crc_controls_expected_risk_on_exchangeable_data():
    rng = np.random.default_rng(5)
    alpha, n, risks = 0.05, 500, []
    for _ in range(400):
        score = rng.random(2 * n)
        loss = (rng.random(2 * n) < 0.25 * score).astype(float)
        tau = crc_threshold(score[:n], loss[:n], alpha)
        risks.append((loss[n:] * (score[n:] <= tau)).mean())
    assert np.mean(risks) <= alpha + 0.003


def test_budgeted_verification():
    verdict = np.array([0, 0, 1, 1], bool)
    fail = np.array([1, 0, 0, 1], bool)
    assert residual_cost(np.zeros(4), verdict, fail, 0, 10, 1) == 11
    assert residual_cost(np.zeros(4), verdict, fail, 4, 10, 1) == 0
    pri = expected_cost(np.array([0.6, 0.1, 0.2, 0.9]), verdict, 10, 1)
    assert residual_cost(pri, verdict, fail, 1, 10, 1) == 1  # the false pass is verified first


# ----------------------------------------------------------------------- simulator
def _single(**kw):
    base = dict(cut_in=0, v_ego=25.0, v_lead=25.0, gap=60.0, t_event=2.0, decel=1.0, ratio=1.0,
                mu=0.9, noise=0.0, latency=0.0, dropout=0.0)
    base.update(kw)
    return np.array([[base[p] for p in PARAMS]])


def test_steady_following_passes():
    out = simulate(_single(), V1, rng=np.random.default_rng(0))
    assert not out["fail"][0] and out["min_gap"][0] > 30


def test_abrupt_stop_on_low_friction_collides():
    x = _single(v_ego=30.0, v_lead=15.0, gap=12.0, decel=9.0, ratio=0.0, mu=0.35, t_event=0.0, latency=0.3)
    out = simulate(x, V1, rng=np.random.default_rng(0))
    assert out["collision"][0] and out["fail"][0]


def test_simulation_is_reproducible_and_friction_matters():
    x = sample_tests(3000, np.random.default_rng(6))
    a = simulate(x, V1, rng=np.random.default_rng(7))
    b = simulate(x, V1, rng=np.random.default_rng(7))
    assert np.array_equal(a["min_gap"], b["min_gap"])
    low = x.copy()
    low[:, IDX["mu"]] = 0.35
    assert simulate(low, V1, rng=np.random.default_rng(7))["fail"].mean() > a["fail"].mean()


def test_software_update_changes_behaviour_and_shift_sampler():
    x = sample_tests(3000, np.random.default_rng(8))
    v1 = simulate(x, V1, rng=np.random.default_rng(9))["fail"]
    v2 = simulate(x, V2, rng=np.random.default_rng(9))["fail"]
    assert v2.mean() > v1.mean()
    xs = sample_tests(1000, np.random.default_rng(10), SHIFTED)
    assert xs[:, IDX["mu"]].max() <= 0.6 and xs[:, IDX["latency"]].min() >= 0.2
