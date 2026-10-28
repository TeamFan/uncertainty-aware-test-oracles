"""Vectorised longitudinal simulator of an ACC/AEB driving function.

The system under test (SUT) is an adaptive cruise control (ACC) with an
automatic emergency braking (AEB) override. A test case fixes a scenario
(lead-vehicle braking or cut-in) and its operating conditions (road friction,
sensor noise, sensor latency, measurement dropout). Executing a test case runs
the closed loop once; sensor noise and dropout are random, so the verdict of a
single execution is itself a random variable (aleatoric uncertainty of the SUT).

All test cases of a batch are simulated in parallel with NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

G = 9.81  # m/s^2

# Columns of a test-case matrix (one row per test case).
PARAMS = [
    "cut_in",      # 0 = lead braking, 1 = cut-in
    "v_ego",       # initial ego speed (m/s), also the ACC set speed
    "v_lead",      # initial speed of the target vehicle (m/s)
    "gap",         # initial gap to the target vehicle (m)
    "t_event",     # onset of lead braking (s)
    "decel",       # requested lead deceleration (m/s^2)
    "ratio",       # lead final speed as a fraction of its initial speed
    "mu",          # road friction coefficient
    "noise",       # gap-measurement noise std (m); rel. speed noise = 0.3 * noise
    "latency",     # sensor latency (s)
    "dropout",     # per-step probability that a measurement is lost
]
IDX = {name: i for i, name in enumerate(PARAMS)}


@dataclass(frozen=True)
class ControllerConfig:
    """Parameters of one software version of the ACC/AEB function."""

    headway: float = 1.5        # desired time gap (s)
    standstill: float = 5.0     # desired gap at standstill (m)
    k_gap: float = 0.23         # gap-error gain
    k_speed: float = 0.74       # relative-speed gain
    k_cruise: float = 0.4       # cruise-speed gain
    a_max: float = 2.0          # max ACC acceleration (m/s^2)
    a_min: float = -3.5         # max ACC (comfort) deceleration (m/s^2)
    ttc_aeb: float = 1.6        # AEB trigger on measured time-to-collision (s)
    a_aeb: float = -9.0         # AEB deceleration request (m/s^2)
    acquisition: float = 0.4    # time to acquire a cut-in target (s)
    version: str = "v1"


V1 = ControllerConfig()
# A later software release tuned for traffic throughput: shorter headway,
# softer relative-speed gain and a later AEB trigger.
V2 = replace(V1, headway=1.2, k_speed=0.55, ttc_aeb=1.25, version="v2")


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.05
    horizon: float = 20.0
    actuator_lag: float = 0.25   # first-order lag of the brake/drive actuator (s)
    safe_gap: float = 2.0        # requirement: minimum gap >= safe_gap (m)
    decel_jitter: float = 0.15   # lead deceleration varies by +/- 15% per execution
    onset_jitter: float = 0.3    # std of the lead braking onset (s)


def simulate(tests: np.ndarray, ctrl: ControllerConfig = V1, sim: SimConfig = SimConfig(),
             rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Execute every test case once.

    Returns a dict of per-test outputs: ``min_gap`` (m, clipped at 0),
    ``collision`` (bool), ``fail`` (bool, requirement violated),
    ``peak_decel`` (m/s^2, comfort / non-functional metric) and
    ``aeb`` (bool, AEB was triggered).
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(tests, dtype=float)
    n = x.shape[0]
    dt = sim.dt
    steps = int(round(sim.horizon / dt))

    cut_in = x[:, IDX["cut_in"]] > 0.5
    v_set = x[:, IDX["v_ego"]].copy()
    mu = x[:, IDX["mu"]]
    a_floor = -mu * G                      # friction-limited deceleration

    # Lead vehicle: brakes at t_event with min(decel, mu g) down to ratio * v_lead.
    # The other road user is not fully controlled by the test: its actual
    # deceleration and braking onset vary between executions.
    decel_draw = x[:, IDX["decel"]] * rng.uniform(1.0 - sim.decel_jitter, 1.0 + sim.decel_jitter, n)
    lead_decel = np.minimum(decel_draw, mu * G)
    lead_target = x[:, IDX["ratio"]] * x[:, IDX["v_lead"]]
    t_event = np.maximum(x[:, IDX["t_event"]] + sim.onset_jitter * rng.standard_normal(n), 0.0)

    noise = x[:, IDX["noise"]]
    lat_steps = np.rint(x[:, IDX["latency"]] / dt).astype(int)
    max_lat = int(lat_steps.max()) if n else 0
    p_drop = x[:, IDX["dropout"]]
    acq_steps = np.where(cut_in, np.rint((ctrl.acquisition + x[:, IDX["latency"]]) / dt), 0).astype(int)

    # State.
    v_e = x[:, IDX["v_ego"]].copy()
    a_e = np.zeros(n)
    v_l = x[:, IDX["v_lead"]].copy()
    gap = x[:, IDX["gap"]].copy()

    hist_gap = np.repeat(gap[None, :], max_lat + 1, axis=0)
    hist_rel = np.repeat((v_l - v_e)[None, :], max_lat + 1, axis=0)
    meas_gap = gap.copy()
    meas_rel = v_l - v_e

    min_gap = gap.copy()
    peak_decel = np.zeros(n)
    aeb_any = np.zeros(n, dtype=bool)
    alive = np.ones(n, dtype=bool)
    cols = np.arange(n)
    lag = dt / sim.actuator_lag

    for k in range(steps):
        t = k * dt
        # Sensor: delayed truth + noise, with random dropout (hold last value).
        slot = k % (max_lat + 1)
        hist_gap[slot] = gap
        hist_rel[slot] = v_l - v_e
        src = (k - lat_steps) % (max_lat + 1)
        delayed_gap = hist_gap[src, cols]
        delayed_rel = hist_rel[src, cols]
        fresh = rng.random(n) >= p_drop
        new_gap = delayed_gap + noise * rng.standard_normal(n)
        new_rel = delayed_rel + 0.3 * noise * rng.standard_normal(n)
        meas_gap = np.where(fresh, new_gap, meas_gap)
        meas_rel = np.where(fresh, new_rel, meas_rel)
        tracked = k >= acq_steps

        # ACC.
        a_cruise = ctrl.k_cruise * (v_set - v_e)
        desired = ctrl.standstill + ctrl.headway * v_e
        a_follow = ctrl.k_gap * (meas_gap - desired) + ctrl.k_speed * meas_rel
        a_cmd = np.where(tracked, np.minimum(a_cruise, a_follow), a_cruise)
        a_cmd = np.clip(a_cmd, ctrl.a_min, ctrl.a_max)

        # AEB override on measured time-to-collision.
        closing = -meas_rel
        ttc = np.where(closing > 0.1, np.maximum(meas_gap, 0.0) / np.maximum(closing, 0.1), np.inf)
        aeb = tracked & (ttc < ctrl.ttc_aeb)
        a_cmd = np.where(aeb, ctrl.a_aeb, a_cmd)
        aeb_any |= aeb & alive

        # Actuator lag and friction limit.
        a_e = a_e + lag * (a_cmd - a_e)
        a_e = np.maximum(a_e, a_floor)
        a_e = np.where(v_e <= 0.0, np.maximum(a_e, 0.0), a_e)

        # Lead vehicle.
        braking = (t >= t_event) & (v_l > lead_target)
        a_l = np.where(braking, -lead_decel, 0.0)

        # Integrate (semi-implicit Euler); frozen after a collision.
        v_e_new = np.maximum(v_e + a_e * dt, 0.0)
        v_l_new = np.maximum(np.where(braking, np.maximum(v_l + a_l * dt, lead_target), v_l), 0.0)
        gap_new = gap + (v_l_new - v_e_new) * dt
        v_e = np.where(alive, v_e_new, v_e)
        v_l = np.where(alive, v_l_new, v_l)
        gap = np.where(alive, gap_new, gap)

        peak_decel = np.where(alive, np.maximum(peak_decel, -a_e), peak_decel)
        min_gap = np.minimum(min_gap, gap)
        alive &= gap > 0.0

    min_gap = np.maximum(min_gap, 0.0)
    collision = min_gap <= 0.0
    return {
        "min_gap": min_gap,
        "collision": collision,
        "fail": min_gap < sim.safe_gap,
        "peak_decel": peak_decel,
        "aeb": aeb_any,
    }


def repeat_fail_probability(tests: np.ndarray, reps: int, ctrl: ControllerConfig = V1,
                            sim: SimConfig = SimConfig(), seed: int = 0) -> np.ndarray:
    """Monte Carlo estimate of P(fail | test case) from ``reps`` executions."""
    rng = np.random.default_rng(seed)
    fails = np.zeros(len(tests))
    for _ in range(reps):
        fails += simulate(tests, ctrl, sim, rng)["fail"]
    return fails / reps
