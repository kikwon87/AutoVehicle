"""The five study tasks set at the end of Lecture 2/3, as runnable experiments.

Each returns a dict of results and prints a short report.  They are here rather
than in the tests because their value is the *numbers*, not a pass/fail: the
point of Study Task 3 is to see where the affordable step runs out, not to
assert that it does.
"""

from __future__ import annotations

import numpy as np

from ..control.geometric import PurePursuit, Stanley
from ..core import se2
from ..core.integrators import convergence_study, max_stable_step, rollout, step
from ..models.dynamic_bicycle import DynamicBicycle, linear_single_track
from ..models.kinematic_bicycle import rear_axle_field
from ..models.params import REFERENCE_VEHICLE as P
from ..world.path import PrimitivePath


def study_1_integrator_convergence(verbose: bool = True) -> dict:
    """Implement Euler, midpoint, Heun and RK4; reproduce the orders 1, 2, 2, 4."""
    L, V, DELTA, T = P.L, 10.0, 0.1, 4.0

    def f(x, u):
        _, _, psi, v = x
        return np.array([v * np.cos(psi), v * np.sin(psi), v / L * np.tan(u[1]), u[0]])

    x0, u = np.array([0.0, 0.0, 0.0, V]), np.array([0.0, DELTA])
    out = {}
    for name in ("euler", "midpoint", "heun", "rk4"):
        steps = [0.4, 0.2, 0.1, 0.05] if name == "rk4" else [0.2, 0.1, 0.05, 0.025, 0.0125]
        r = convergence_study(f, x0, u, T, steps, name)
        out[name] = {
            "observed_order": r["observed_order"],
            "nominal_order": r["nominal_order"],
            "error_at_h_0.1": float(np.interp(0.1, r["h"][::-1], r["error"][::-1])),
        }
    if verbose:
        print("Study 1 -- integrator convergence (kinematic bicycle, 4 s, delta = 0.1 rad)")
        for k, v in out.items():
            print(f"  {k:9s} observed order {v['observed_order']:5.2f} "
                  f"(nominal {v['nominal_order']})   global error at h = 0.1 s: "
                  f"{v['error_at_h_0.1']:.3e} m")
    return out


def study_2_constant_twist(verbose: bool = True) -> dict:
    """``exp`` on SE(2) versus Euler on the chart; recover the 0.84 m drift."""
    v, w, h, T = 10.0, 0.5, 0.1, 4.0
    xi = np.array([v, 0.0, w])
    exact = se2.unpose(se2.exp(T * xi))

    g = np.eye(3)
    for _ in range(int(T / h)):
        g = g @ se2.exp(h * xi)
    group_err = float(np.linalg.norm(se2.unpose(g)[:2] - exact[:2]))

    x = np.zeros(3)
    for _ in range(int(T / h)):
        x = x + h * np.array([v * np.cos(x[2]), v * np.sin(x[2]), w])
    euler_err = float(np.linalg.norm(x[:2] - exact[:2]))
    one_step = float(np.linalg.norm((h * xi)[:2] - se2.unpose(se2.exp(h * xi))[:2]))

    out = {"radius": v / w, "group_step_error": group_err,
           "euler_chart_error": euler_err, "one_step_chord_vs_arc": one_step}
    if verbose:
        print(f"Study 2 -- constant twist, R = {out['radius']:.1f} m, {T:.0f} s at h = {h} s")
        print(f"  Lie-group step   {group_err:.2e} m   (machine precision)")
        print(f"  Euler on (X,Y,psi) {euler_err:.3f} m   (systematic outward drift)")
        print(f"  one step, chord vs arc {one_step*100:.2f} cm")
    return out


def study_3_stability(h: float = 0.1, verbose: bool = True) -> dict:
    """Build ``A(V)``, take its spectrum, and find the Euler and RK4 speed limits."""
    speeds = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40], dtype=float)
    rows = []
    for V in speeds:
        A, _ = linear_single_track(V, P)
        eig = np.linalg.eigvals(A)
        rows.append({
            "V": float(V),
            "eigenvalues": eig.tolist(),
            "h_max_euler": max_stable_step(eig, "euler"),
            "h_max_rk4": max_stable_step(eig, "rk4"),
        })
    v_euler = next((r["V"] for r in rows if r["h_max_euler"] >= h), None)
    v_rk4 = next((r["V"] for r in rows if r["h_max_rk4"] >= h), None)
    out = {"h": h, "rows": rows, "euler_min_speed": v_euler, "rk4_min_speed": v_rk4}
    if verbose:
        print(f"Study 3 -- stability of the linear single-track model at h = {h} s")
        for r in rows:
            ok_e = "ok " if r["h_max_euler"] >= h else "NO "
            ok_r = "ok " if r["h_max_rk4"] >= h else "NO "
            print(f"  V = {r['V']:5.1f} m/s  h_max: euler {r['h_max_euler']:.4f} {ok_e} "
                  f"rk4 {r['h_max_rk4']:.4f} {ok_r}")
        print(f"  Euler needs V >= {v_euler} m/s, RK4 needs V >= {v_rk4} m/s at this step.")
        print("  This is a failure of the model near standstill, not of the integrator:")
        print("  the remedy is a low-speed-valid model, not a smaller step.")
    return out


def study_4_tracking_baselines(verbose: bool = True) -> dict:
    """Pure pursuit and Stanley with explicit signs; cross-track and steering rate."""
    R, V, h, T = 200.0, 25.0, 0.01, 12.0
    path = PrimitivePath.chain(0, 0, 0, [("arc", R * np.pi / 2, 1 / R)])
    out = {}
    for plant in ("kinematic", "dynamic"):
        for name, ctl in [
            ("pure_pursuit", PurePursuit(P)),
            ("pure_pursuit_ff", PurePursuit(P, use_feedforward=True)),
            ("stanley", Stanley(P)),
            ("stanley_ff", Stanley(P, use_feedforward=True)),
        ]:
            if plant == "kinematic":
                f = rear_axle_field(P)
                x = np.array([0.0, 0.0, 0.0, V])
                get = lambda z: (z[0], z[1], z[2], z[3])  # noqa: E731
            else:
                m = DynamicBicycle(P, tire_model="linear")
                f = m.field()
                x = m.initial_state(v=V)
                get = lambda z: (  # noqa: E731
                    z[0] - P.l_r * np.cos(z[2]), z[1] - P.l_r * np.sin(z[2]), z[2], z[3]
                )
            s, prev, errs, rates = 0.0, 0.0, [], []
            for k in range(int(T / h)):
                X, Y, psi, v = get(x)
                delta, info = ctl(X, Y, psi, v, path, s)
                s = info["s"]
                rates.append(abs(delta - prev) / h)
                prev = delta
                if plant == "kinematic":
                    x = step(f, x, np.array([0.0, delta]), h, "rk4")
                else:
                    x = m.sanitize(step(f, x, np.array([(V - x[3]) * 1.5, delta]), h, "rk4"))
                errs.append(path.lateral_offset(X, Y, s))
            e = np.array(errs)
            out[f"{plant}/{name}"] = {
                "steady_state_e_y": float(e[-1]),
                "rms_e_y": float(np.sqrt(np.mean(e[len(e) // 2:] ** 2))),
                "max_steer_rate": float(max(rates[1:])),
            }
    if verbose:
        print(f"Study 4 -- tracking baselines, R = {R:.0f} m at {V:.0f} m/s "
              f"(a_y = {V**2/R:.2f} m/s^2)")
        print(f"  required delta_ss = {P.steady_state_steer(1/R, V):.5f} rad "
              f"= geometry {np.arctan(P.L/R):.5f} + force {P.understeer_gradient*V*V/R:.5f}")
        for k, v in out.items():
            print(f"  {k:28s} steady e_y {v['steady_state_e_y']:+7.4f} m   "
                  f"rms {v['rms_e_y']:.4f} m   peak steer rate {v['max_steer_rate']:.3f} rad/s")
        print("  On the tireless kinematic model the feedforward hurts; on the vehicle")
        print("  that has tires it removes most of the steady-state error.")
    return out


def study_5_model_contract(verbose: bool = True) -> dict:
    """State the coordinates, regime, integrator, h, N, parameters and inter-sample rule."""
    from ..control.mpc import MPCConfig

    cfg = MPCConfig()
    contract = {
        "manoeuvre": "unprotected left turn: brake, low-speed turn, accelerate",
        "coordinates": "world chart at the rear axle, [X, Y, psi, v, delta]; "
                       "lane constraints in the Frenet frame of the route",
        "model_regime": "kinematic bicycle with the understeer correction "
                        "psi' = v tan(delta) / (L + K_us v^2); the plant is the "
                        "nonlinear dynamic bicycle with a saturating tire",
        "integrator": "classical RK4 on the prediction model; RK4 on the plant "
                      "with the control held (zero-order hold)",
        "h": cfg.dt,
        "N": cfg.horizon,
        "horizon_s": cfg.dt * cfg.horizon,
        "control_period_s": 0.1,
        "plant_step_s": 0.02,
        "parameters": {
            "m": P.m, "I_z": P.I_z, "l_f": P.l_f, "l_r": P.l_r,
            "C_f": P.C_f, "C_r": P.C_r, "mu": P.mu,
            "K_us": P.understeer_gradient, "v_ch": P.characteristic_speed,
        },
        "input_limits": {
            "a": (P.actuator.a_min, P.actuator.a_max),
            "delta_rate": (-P.actuator.delta_rate_max, P.actuator.delta_rate_max),
        },
        "state_constraints": [
            "|delta| <= 0.95 delta_max",
            "0 <= v <= v_max",
            "|v^2 tan(delta) / (L + K_us v^2)| <= 0.5 mu g",
            "lane corridor as two affine half-spaces per node",
            "obstacle clearance as oriented ellipses between multi-circle covers",
            "stop line as one half-space on the front bumper",
        ],
        "inter_sample_treatment": (
            "constraints hold at the nodes only. Obstacles are tightened by half "
            "the distance they travel between samples; the lane corridor is "
            "tightened by the vehicle half-width. Neither is a proof of "
            "inter-sample feasibility."
        ),
        "transcription": "single shooting (iLQR), dynamics feasible by construction; "
                         "path constraints by augmented Lagrangian",
        "derivatives": "analytic Jacobians of the RK4 step (discretize, then linearize)",
    }
    if verbose:
        print("Study 5 -- model contract")
        for k, v in contract.items():
            if isinstance(v, (dict, list)):
                print(f"  {k}:")
                items = v.items() if isinstance(v, dict) else enumerate(v)
                for kk, vv in items:
                    print(f"      {kk}: {vv}" if isinstance(v, dict) else f"      - {vv}")
            else:
                print(f"  {k}: {v}")
    return contract


STUDIES = {
    "1": study_1_integrator_convergence,
    "2": study_2_constant_twist,
    "3": study_3_stability,
    "4": study_4_tracking_baselines,
    "5": study_5_model_contract,
}


def run_all_studies(verbose: bool = True) -> dict:
    out = {}
    for k, fn in STUDIES.items():
        out[k] = fn(verbose=verbose)
        if verbose:
            print()
    return out
