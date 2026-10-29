"""Run every experiment of the study and write results, figures and report numbers.

Usage:  python scripts/run_experiments.py   (about 40 seconds on 2 CPU cores)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqoracle.oracles import GBEnsemble, MLPEnsemble, Novelty, QuantileRegressor  # noqa: E402
from uqoracle.risk_control import crc_threshold, expected_cost, residual_cost  # noqa: E402
from uqoracle.scenarios import SHIFTED, sample_tests  # noqa: E402
from uqoracle.simulator import V1, V2, repeat_fail_probability, simulate  # noqa: E402
from uqoracle.uncertainty import (auroc, aurc, binary_entropy, ece, ensemble_scores,  # noqa: E402
                                  risk_coverage, single_pass_scores)

RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = 20261028
N_TRAIN, N_FIT, N_POOL, N_SHIFT = 4000, 3000, 6000, 5000
N_CAL = 2000
N_REPEAT_SPLITS = 200
ALPHAS = [0.005, 0.01, 0.02]
ALPHA_MAIN = 0.01
N_RECAL = 300
C_FP, C_FF = 10.0, 1.0
K_REPS = 20
N_BOOT = 1000

t0 = time.time()
rng = np.random.default_rng(SEED)


def execute(x, ctrl):
    return simulate(x, ctrl, rng=rng)


# ----------------------------------------------------------------------------- data
x_train = sample_tests(N_TRAIN, rng); o_train = execute(x_train, V1)
x_fit = sample_tests(N_FIT, rng); o_fit = execute(x_fit, V1)
x_s0 = sample_tests(N_POOL, rng); o_s0 = execute(x_s0, V1)
x_s1 = sample_tests(N_SHIFT, rng, SHIFTED); o_s1 = execute(x_s1, V1)
x_s2 = sample_tests(N_SHIFT, rng); o_s2 = execute(x_s2, V2)
SETS = {"S0": (x_s0, o_s0), "S1": (x_s1, o_s1), "S2": (x_s2, o_s2)}

# ----------------------------------------------------------------------------- oracles
mlp = MLPEnsemble(5, seed=1).fit(x_train, o_train["fail"])
gb = GBEnsemble(5, seed=2).fit(x_train, o_train["fail"])
nov = Novelty(k=10).fit(x_train)


def raw_scores(x):
    mp = mlp.member_proba(x)
    p = mp.mean(axis=0)
    pg = gb.proba(x)
    es = ensemble_scores(mp)
    sp = single_pass_scores(p)
    return {
        "p": p, "p_single": mp[0], "p_gb": pg,
        "one_minus_msp": sp["one_minus_msp"], "gini": sp["gini"], "confidence": es["total_entropy"],
        "aleatoric": es["aleatoric"], "mutual_info": es["mutual_info"],
        "disagreement": np.abs(p - pg), "novelty": nov.score(x),
        "single_entropy": binary_entropy(mp[0]),
    }


STACK_COLS = ["confidence", "mutual_info", "disagreement", "novelty"]

fit = raw_scores(x_fit)
y_fit = o_fit["fail"]
err_fit = (fit["p"] > 0.5) != y_fit
stack_feats = lambda s: np.column_stack([np.log(s[c] + 1e-6) for c in STACK_COLS])  # noqa: E731
stacker = LogisticRegression(C=1.0, max_iter=1000).fit(stack_feats(fit), err_fit)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(fit["p"], y_fit)


def all_scores(x):
    s = raw_scores(x)
    s["stacked"] = stacker.predict_proba(stack_feats(s))[:, 1]
    s["p_cal"] = iso.predict(s["p"])
    return s


SC = {k: all_scores(v[0]) for k, v in SETS.items()}
SIGNALS = ["confidence", "mutual_info", "disagreement", "novelty", "stacked"]
LABEL = {"confidence": "Confidence (entropy of mean)", "mutual_info": "Ensemble mutual information",
         "disagreement": "Cross-family disagreement", "novelty": "Input novelty (kNN)",
         "stacked": "Stacked error predictor", "single": "Single network, one pass"}

summary: dict = {"config": {"seed": SEED, "n_train": N_TRAIN, "n_fit": N_FIT, "n_pool_s0": N_POOL,
                            "n_shift": N_SHIFT, "n_cal": N_CAL, "splits": N_REPEAT_SPLITS,
                            "alphas": ALPHAS, "n_recal": N_RECAL, "c_false_pass": C_FP,
                            "c_false_fail": C_FF, "reps_flakiness": K_REPS}}

# ----------------------------------------------------------------------------- E1: error detection
e1 = {}
brng = np.random.default_rng(SEED + 1)
for name, (x, o) in SETS.items():
    s = SC[name]
    y = o["fail"]
    pred = s["p"] > 0.5
    err = pred != y
    row = {"fail_rate": float(y.mean()), "error_rate": float(err.mean()),
           "false_pass_rate": float(((~pred) & y).mean()), "false_fail_rate": float((pred & ~y).mean()),
           "error_rate_single": float(((s["p_single"] > 0.5) != y).mean()),
           "error_rate_gb": float(((s["p_gb"] > 0.5) != y).mean()),
           "ece": ece(s["p"], y), "ece_isotonic": ece(s["p_cal"], y),
           "mean_mutual_info": float(s["mutual_info"].mean()), "mean_novelty": float(s["novelty"].mean()),
           "signals": {}}
    for sig in SIGNALS + ["one_minus_msp", "gini"]:
        a = auroc(s[sig], err)
        boots = []
        for _ in range(N_BOOT if sig in SIGNALS else 0):
            b = brng.integers(0, len(err), len(err))
            boots.append(auroc(s[sig][b], err[b]))
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        row["signals"][sig] = {"auroc": a, "auroc_ci": [float(lo), float(hi)], "aurc": aurc(s[sig], err)}
    err_single = (s["p_single"] > 0.5) != y
    row["signals"]["single"] = {"auroc": auroc(s["single_entropy"], err_single),
                                "aurc": aurc(s["single_entropy"], err_single)}
    e1[name] = row
summary["E1_error_detection"] = e1

# ----------------------------------------------------------------------------- E2: risk control
def fp_loss(s, y, which="p"):
    return ((s[which] <= 0.5) & y).astype(float)


def run_crc(score_key, alpha, split_rng, recal_n=N_RECAL):
    s0, y0 = SC["S0"], o_s0["fail"]
    loss0 = fp_loss(s0, y0)
    out = {k: [] for k in ["S0_risk", "S0_auto", "S1_risk", "S1_auto", "S2_risk", "S2_auto",
                           "S1_recal_risk", "S1_recal_auto", "S2_recal_risk", "S2_recal_auto"]}
    for _ in range(N_REPEAT_SPLITS):
        perm = split_rng.permutation(N_POOL)
        cal, test = perm[:N_CAL], perm[N_CAL:]
        tau = crc_threshold(s0[score_key][cal], loss0[cal], alpha)
        acc = s0[score_key][test] <= tau
        out["S0_risk"].append(float((acc * loss0[test]).mean()))
        out["S0_auto"].append(float(acc.mean()))
        for sh in ["S1", "S2"]:
            s, y = SC[sh], SETS[sh][1]["fail"]
            loss = fp_loss(s, y)
            acc = s[score_key] <= tau
            out[f"{sh}_risk"].append(float((acc * loss).mean()))
            out[f"{sh}_auto"].append(float(acc.mean()))
            p2 = split_rng.permutation(len(y))
            rc, rt = p2[:recal_n], p2[recal_n:]
            tau_r = crc_threshold(s[score_key][rc], loss[rc], alpha)
            acc_r = s[score_key][rt] <= tau_r
            out[f"{sh}_recal_risk"].append(float((acc_r * loss[rt]).mean()))
            out[f"{sh}_recal_auto"].append(float(acc_r.mean()))
    return {k: {"mean": float(np.mean(v)), "violation_share": None} for k, v in out.items()}, out


e2 = {}
raw_e2 = {}
for sig in SIGNALS:
    e2[sig] = {}
    for alpha in ALPHAS:
        agg, raw = run_crc(sig, alpha, np.random.default_rng(SEED + 7))
        for k in agg:
            if k.endswith("risk"):
                agg[k]["violation_share"] = float(np.mean(np.array(raw[k]) > alpha))
        e2[sig][str(alpha)] = agg
        raw_e2[(sig, alpha)] = raw
summary["E2_risk_control"] = e2

# ----------------------------------------------------------------------------- E3: budgeted verification
budgets = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
e3 = {}
prng = np.random.default_rng(SEED + 11)
for name in ["S0", "S1", "S2"]:
    s, y = SC[name], SETS[name][1]["fail"]
    verdict = s["p"] > 0.5
    base = residual_cost(np.zeros(len(y)), verdict, y, 0, C_FP, C_FF)
    pol = {"random": prng.random(len(y)),
           "confidence": s["confidence"],
           "stacked": s["stacked"],
           "expected_cost": expected_cost(s["p_cal"], verdict, C_FP, C_FF)}
    e3[name] = {"cost_no_verification": base, "curves": {}}
    for pname, pr in pol.items():
        e3[name]["curves"][pname] = [residual_cost(pr, verdict, y, int(round(b * len(y))), C_FP, C_FF) / base
                                     for b in budgets]
summary["E3_budget"] = {"budgets": budgets, "results": e3}

# ----------------------------------------------------------------------------- E4: non-functional property
qr = QuantileRegressor(alpha=0.1, seed=3).fit(x_train, o_train["peak_decel"])
lo_f, hi_f = qr.predict(x_fit)
conf_score = np.maximum(lo_f - o_fit["peak_decel"], o_fit["peak_decel"] - hi_f)
n_f = len(conf_score)
qhat = np.quantile(conf_score, min(1.0, np.ceil((n_f + 1) * 0.9) / n_f), method="higher")
e4 = {"qhat": float(qhat)}
for name, (x, o) in SETS.items():
    lo, hi = qr.predict(x)
    lo, hi = lo - qhat, hi + qhat
    cover = (o["peak_decel"] >= lo) & (o["peak_decel"] <= hi)
    width = hi - lo
    margin = np.abs(o["min_gap"] - 2.0)
    s = SC[name]
    e4[name] = {"coverage": float(cover.mean()), "mean_width": float(width.mean()),
                "spearman_unc_width": float(stats.spearmanr(s["confidence"], width)[0]),
                "spearman_unc_margin": float(stats.spearmanr(s["confidence"], margin)[0]),
                "coverage_top_decile_unc": float(cover[s["confidence"] >= np.quantile(s["confidence"], 0.9)].mean())}
summary["E4_non_functional"] = e4

# ----------------------------------------------------------------------------- E5: aleatoric vs flakiness
sub = np.arange(3000)
p_true = repeat_fail_probability(x_s0[sub], K_REPS, V1, seed=SEED + 21)
flaky = (p_true > 0) & (p_true < 1)
h_true = binary_entropy(p_true)
s = SC["S0"]
e5 = {"share_flaky": float(flaky.mean()),
      "bayes_error": float(np.minimum(p_true, 1 - p_true).mean()),
      "spearman_aleatoric_true_entropy": float(stats.spearmanr(s["aleatoric"][sub], h_true)[0]),
      "spearman_mi_true_entropy": float(stats.spearmanr(s["mutual_info"][sub], h_true)[0]),
      "auroc_aleatoric_flaky": auroc(s["aleatoric"][sub], flaky),
      "auroc_mi_flaky": auroc(s["mutual_info"][sub], flaky),
      "mean_mi_ratio_S1_S0": float(SC["S1"]["mutual_info"].mean() / SC["S0"]["mutual_info"].mean()),
      "mean_mi_ratio_S2_S0": float(SC["S2"]["mutual_info"].mean() / SC["S0"]["mutual_info"].mean())}
summary["E5_aleatoric"] = e5

# ----------------------------------------------------------------------------- E6: random audit after a software update
arng = np.random.default_rng(SEED + 31)
p0 = e1["S0"]["error_rate"]
e6 = {"nominal_error": p0, "power": {}}
for n_aud in [100, 200, 400, 800]:
    rej_s2 = rej_s0 = 0
    for _ in range(2000):
        idx = arng.choice(N_SHIFT, n_aud, replace=False)
        k = int(((SC["S2"]["p"][idx] > 0.5) != o_s2["fail"][idx]).sum())
        rej_s2 += stats.binomtest(k, n_aud, p0, alternative="greater").pvalue < 0.05
        idx0 = arng.choice(N_POOL, n_aud, replace=False)
        k0 = int(((SC["S0"]["p"][idx0] > 0.5) != o_s0["fail"][idx0]).sum())
        rej_s0 += stats.binomtest(k0, n_aud, p0, alternative="greater").pvalue < 0.05
    e6["power"][str(n_aud)] = {"detect_S2": rej_s2 / 2000, "false_alarm_S0": rej_s0 / 2000}
summary["E6_audit"] = e6
summary["runtime_s"] = round(time.time() - t0, 1)

(RES / "summary.json").write_text(json.dumps(summary, indent=2))

# ----------------------------------------------------------------------------- figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"confidence": "#1b6f8a", "mutual_info": "#c1662b", "disagreement": "#6a8f3a", "novelty": "#8a6fb0",
       "stacked": "#222222", "random": "#9a9a9a", "expected_cost": "#b8333a"}

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95), sharey=True)
for ax, name in zip(axes, ["S0", "S1", "S2"]):
    s, y = SC[name], SETS[name][1]["fail"]
    err = (s["p"] > 0.5) != y
    for sig in SIGNALS:
        cov, risk = risk_coverage(s[sig], err)
        ax.plot(cov, risk, color=COL[sig], lw=1.2, label=LABEL[sig])
    ax.axhline(err.mean(), color="#bbbbbb", lw=0.8, ls=":")
    ax.set_title({"S0": "S0: nominal ODD", "S1": "S1: low friction, slow sensing", "S2": "S2: software update"}[name], fontsize=8.5)
    ax.set_xlabel("coverage (share accepted)")
axes[0].set_ylabel("error rate among accepted")
axes[0].set_ylim(0, 0.1)
handles, labs = axes[0].get_legend_handles_labels()
fig.legend(handles, labs, fontsize=6.5, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(FIG / "risk_coverage.pdf")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
ax = axes[0]
labels = ["S0", "S1\n(S0 cal.)", "S1\n(recal.)", "S2\n(S0 cal.)", "S2\n(recal.)"]
keys = ["S0_risk", "S1_risk", "S1_recal_risk", "S2_risk", "S2_recal_risk"]
w = 0.16
for j, sig in enumerate(["confidence", "mutual_info", "stacked"]):
    raw = raw_e2[(sig, ALPHA_MAIN)]
    vals = [np.mean(raw[k]) for k in keys]
    ax.bar(np.arange(5) + (j - 1) * w * 1.2, vals, w, color=COL[sig], label=LABEL[sig])
ax.axhline(ALPHA_MAIN, color="#b8333a", lw=1, ls="--", label=r"target $\alpha$")
ax.set_xticks(np.arange(5))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("false passes per test")
ax.legend(fontsize=6.5, frameon=False)
ax = axes[1]
for j, sig in enumerate(["confidence", "mutual_info", "stacked"]):
    raw = raw_e2[(sig, ALPHA_MAIN)]
    vals = [np.mean(raw[k.replace("risk", "auto")]) for k in keys]
    ax.bar(np.arange(5) + (j - 1) * w * 1.2, vals, w, color=COL[sig])
ax.set_xticks(np.arange(5))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("share of verdicts accepted")
fig.tight_layout()
fig.savefig(FIG / "risk_control.pdf")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, name in zip(axes, ["S0", "S1"]):
    for pname in ["random", "confidence", "stacked", "expected_cost"]:
        ax.plot(np.array(budgets) * 100, e3[name]["curves"][pname], marker="o", ms=2.5, lw=1.2,
                color=COL[pname], label={"random": "Random", "confidence": "Confidence",
                                         "stacked": "Stacked error predictor",
                                         "expected_cost": "Expected cost (calibrated)"}[pname])
    ax.set_xlabel("simulation budget (% of test cases)")
    ax.set_title({"S0": "S0: nominal ODD", "S1": "S1: shifted ODD"}[name], fontsize=8.5)
axes[0].set_ylabel("residual cost / no verification")
axes[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "budget.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 2.5))
s = SC["S0"]
hb = ax.hexbin(o_s0["min_gap"] - 2.0, s["confidence"], gridsize=40, bins="log", cmap="Blues",
               extent=(-2, 30, 0, 0.7), mincnt=1)
ax.axvline(0, color="#b8333a", lw=0.8, ls="--")
ax.set_xlabel("safety margin: min gap $-$ 2 m (m)")
ax.set_ylabel("verdict uncertainty (entropy)")
fig.colorbar(hb, ax=ax, label="test cases (log)")
fig.tight_layout()
fig.savefig(FIG / "margin.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------- report numbers
def pct(v, d=1):
    return f"{100 * v:.{d}f}"


nums = {
    "FailSZero": pct(e1["S0"]["fail_rate"]), "FailSOne": pct(e1["S1"]["fail_rate"]), "FailSTwo": pct(e1["S2"]["fail_rate"]),
    "ErrSZero": pct(e1["S0"]["error_rate"]), "ErrSOne": pct(e1["S1"]["error_rate"]), "ErrSTwo": pct(e1["S2"]["error_rate"]),
    "ErrSingle": pct(e1["S0"]["error_rate_single"]), "ErrGB": pct(e1["S0"]["error_rate_gb"]),
    "EceSZero": f"{e1['S0']['ece']:.3f}", "EceSOne": f"{e1['S1']['ece']:.3f}", "EceSTwo": f"{e1['S2']['ece']:.3f}",
    "MIRatioOne": f"{e5['mean_mi_ratio_S1_S0']:.1f}", "MIRatioTwo": f"{e5['mean_mi_ratio_S2_S0']:.2f}",
    "Flaky": pct(e5["share_flaky"]), "BayesErr": pct(e5["bayes_error"]),
    "AurocAleFlaky": f"{e5['auroc_aleatoric_flaky']:.2f}", "AurocMIFlaky": f"{e5['auroc_mi_flaky']:.2f}",
    "CovSZero": pct(e4["S0"]["coverage"]), "CovSOne": pct(e4["S1"]["coverage"]), "CovSTwo": pct(e4["S2"]["coverage"]),
    "RhoMargin": f"{e4['S0']['spearman_unc_margin']:.2f}", "RhoWidth": f"{e4['S0']['spearman_unc_width']:.2f}",
    "NTotal": f"{N_TRAIN + N_FIT + N_POOL + 2 * N_SHIFT:,}", "NReps": f"{3000 * K_REPS:,}",
}
for sig, short in [("confidence", "Conf"), ("mutual_info", "MI"), ("stacked", "Stack"),
                   ("disagreement", "Dis"), ("novelty", "Nov")]:
    r = e2[sig][str(ALPHA_MAIN)]
    nums[f"Auto{short}"] = pct(r["S0_auto"]["mean"])
    nums[f"Risk{short}"] = f"{100 * r['S0_risk']['mean']:.2f}"
    nums[f"RiskOne{short}"] = f"{100 * r['S1_risk']['mean']:.2f}"
    nums[f"RiskTwo{short}"] = f"{100 * r['S2_risk']['mean']:.2f}"
    nums[f"RecalOne{short}"] = f"{100 * r['S1_recal_risk']['mean']:.2f}"
    nums[f"RecalTwo{short}"] = f"{100 * r['S2_recal_risk']['mean']:.2f}"
    nums[f"AutoRecalOne{short}"] = pct(r["S1_recal_auto"]["mean"])
    nums[f"AutoRecalTwo{short}"] = pct(r["S2_recal_auto"]["mean"])
    for sh, tag in [("S0", "Zero"), ("S1", "One"), ("S2", "Two")]:
        nums[f"Auroc{tag}{short}"] = f"{e1[sh]['signals'][sig]['auroc']:.3f}"
for sh, tag in [("S0", "Zero"), ("S1", "One")]:
    c = e3[sh]["curves"]
    i20 = budgets.index(0.20)
    nums[f"BudRand{tag}"] = pct(1 - c["random"][i20], 0)
    nums[f"BudConf{tag}"] = pct(1 - c["confidence"][i20], 0)
    nums[f"BudCost{tag}"] = pct(1 - c["expected_cost"][i20], 0)
    nums[f"BudStack{tag}"] = pct(1 - c["stacked"][i20], 0)
nums["AuditDetect"] = pct(e6["power"]["400"]["detect_S2"], 0)
nums["AuditFalse"] = pct(e6["power"]["400"]["false_alarm_S0"], 1)
nums["AuditDetectTwoHundred"] = pct(e6["power"]["200"]["detect_S2"], 0)
lines = [f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in nums.items()]
(ROOT / "report" / "numbers.tex").write_text("\n".join(lines) + "\n")
print(json.dumps({k: v for k, v in nums.items()}, indent=1))
print("runtime", summary["runtime_s"], "s")

# ----------------------------------------------------------------------------- report tables
TAB = ROOT / "report" / "tables"
TAB.mkdir(exist_ok=True)
rows = []
for sig in ["confidence", "mutual_info", "disagreement", "novelty", "stacked"]:
    cells = []
    for sh in ["S0", "S1", "S2"]:
        r = e1[sh]["signals"][sig]
        cells.append(f"{r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]")
    rows.append(f"{LABEL[sig]} & " + " & ".join(cells) + r" \\")
r_single = [f"{e1[sh]['signals']['single']['auroc']:.3f}" for sh in ["S0", "S1", "S2"]]
rows.append(r"\midrule" + "\n" + f"{LABEL['single']}$^\\ast$ & " + " & ".join(r_single) + r" \\")
(TAB / "e1.tex").write_text("\n".join(rows) + "\n")

rows = []
for sig in ["confidence", "mutual_info", "disagreement", "novelty", "stacked"]:
    r = e2[sig][str(ALPHA_MAIN)]
    f = lambda k: f"{100 * r[k]['mean']:.2f}"  # noqa: E731
    a = lambda k: f"{100 * r[k]['mean']:.0f}"  # noqa: E731
    rows.append(f"{LABEL[sig]} & {f('S0_risk')} & {a('S0_auto')} & {f('S1_risk')} & {f('S1_recal_risk')} & "
                f"{a('S1_recal_auto')} & {f('S2_risk')} & {f('S2_recal_risk')} & {a('S2_recal_auto')} \\\\")
(TAB / "e2.tex").write_text("\n".join(rows) + "\n")
