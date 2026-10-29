# When Can a Learned Test Oracle Be Trusted?

**Uncertainty-guided acceptance of verdicts for an automated driving function**

Independent technical study, October 2026 (Xiangyu Luo).
Report: [`report/main.pdf`](report/main.pdf) (5 pages).

A learned test oracle predicts whether a test case of an autonomous
cyber-physical system passes its safety requirement without running the
simulation. Its verdicts are cheap but fallible. This study asks when such a
verdict can be **accepted** and when the test must be **executed**, and treats
the question as a risk-controlled decision:

1. **Q1 – Detection.** Which uncertainty scores identify wrong verdicts, and what
   do multi-inference scores (ensembles, a second model family) add to
   single-pass confidence?
2. **Q2 – Guarantees.** Can an acceptance threshold bound the rate of *false
   passes*, and does the bound survive a shift of the operational design domain
   (ODD) or a software update of the system under test (SUT)?
3. **Q3 – Budget.** With a limited simulation budget, which verdicts should be
   re-executed?
4. **Q4 – Non-functional properties.** How does verdict uncertainty relate to
   the safety margin and to a comfort metric of the execution?

## Testbed

* **SUT:** adaptive cruise control with automatic emergency braking, simulated
  longitudinally in NumPy (0.05 s steps, 20 s), with friction-limited braking,
  actuator lag, delayed and noisy perception with dropout
  ([`simulator.py`](src/uqoracle/simulator.py)).
* **Test cases:** 11 parameters (lead braking or cut-in, speeds, gap, braking
  onset and strength, friction, sensor noise, latency, dropout). Requirement:
  minimum gap ≥ 2 m. The other road user varies between executions, so 6.2 % of
  nominal test cases are flaky
  (estimated irreducible classification error 1.2 %).
* **Shifts:** S1 = wet/snowy roads and slower sensing (ODD shift);
  S2 = software update of the controller (shorter headway, later AEB) on the
  same test distribution.
* **Oracle:** deep ensemble of 5 MLPs (bootstrap), plus 5 gradient-boosted
  tree classifiers as an independent model family.
* **Scores:** confidence (entropy of the ensemble mean), ensemble mutual
  information, cross-family disagreement, input novelty (kNN), and a stacked
  logistic error predictor fitted on a separate set.
* **Acceptance:** conformal risk control (Angelopoulos et al., ICLR 2024) of the
  expected number of accepted false passes per test case at α = 1 %
  (a marginal guarantee under exchangeability); budgeted re-execution by estimated
  expected cost (false pass 10, false fail 1); random audit with a binomial
  test after a software update.

## Main results (23,000 simulated executions, 200 calibration splits)

| | S0 nominal | S1 ODD shift | S2 software update |
|---|---|---|---|
| Failure rate of executions | 18.0 % | 28.8 % | 24.6 % |
| Oracle error rate (ensemble) | 4.8 % | 6.2 % | 8.3 % |
| AUROC for wrong verdicts: confidence / stacked | 0.940 / 0.939 | 0.923 / 0.927 | 0.894 / 0.890 |
| Mean mutual information relative to S0 | 1 | 1.9× | 0.90× |
| False passes per test, threshold calibrated on S0 (target 1 %): confidence / stacked | 0.98 / 0.97 % | **1.64** / 0.93 % | **4.65 / 4.29 %** |
| Share of verdicts accepted (confidence, S0 threshold) | 90.8 % | 88 % | 91 % |
| same, recalibrated on 300 shifted executions: risk / accepted (stacked) | – | 0.92 % / 81 % | 1.00 % / 70 % |

* For binary verdicts, maximum softmax probability, entropy, Gini impurity and
  margin are strictly decreasing in |p − 1/2| and rank test cases identically
  (proved in the report, checked in the tests).
* After the software update, all five acceptance rules calibrated on the nominal
  domain exceed the 1 % target (2.98–4.65 % false passes per test case), although
  the scores still rank errors well (AUROC up to 0.894): a frozen oracle scoring
  unchanged inputs cannot see a change of the system that produced the labels.
  A random audit of 400 executions detects the update with an estimated 90 %
  power (4.8 % false alarms on the unchanged system).
* Re-executing the 20 % of verdicts with the highest calibrated expected cost
  removes 94 % of the misjudgement cost on S0 and 88 % on S1
  (random selection: 19 % and 17 %).
* Verdict uncertainty concentrates near the requirement boundary (Spearman
  −0.64 with the absolute safety margin); conformal 90 % intervals for peak
  deceleration cover 90.5 % nominally and 84.9 % / 83.7 % under S1 / S2.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/run_experiments.py   # ~40 s on 2 CPU cores; writes results/, report/figures/, report/numbers.tex
pytest -q                            # 12 tests
cd report && latexmk -pdf main.tex
```

## Layout

```
src/uqoracle/simulator.py      ACC/AEB closed-loop simulator, software versions V1/V2
src/uqoracle/scenarios.py      nominal and shifted test-case generators
src/uqoracle/oracles.py        MLP and gradient-boosting ensembles, kNN novelty, quantile regressor
src/uqoracle/uncertainty.py    single-pass and ensemble scores, AUROC, AURC, ECE
src/uqoracle/risk_control.py   conformal risk control threshold, expected-cost ranking
scripts/run_experiments.py     all experiments, figures and report numbers
results/summary.json           every reported number
tests/test_core.py             unit tests
```

## Scope

The SUT is a longitudinal model with two scenario families, and the learned
oracle is a tabular classifier, not a large language or vision-language model.
The acceptance rule, loss and audit need only a verdict and a score, so the
protocol carries over to foundation-model oracles whose multi-inference scores
come from repeated sampling or several models.
