# avsim — 2-D autonomous driving simulator

A modular, graduate-level simulator for **autonomy behaviours**: vehicle
dynamics, a 2-D world with signalized intersections and other traffic, simple
vision-based perception, trajectory generation, path following, MPC-based
optimal control, and a KPI harness that says whether a run was acceptable.

Built against *Introduction to Autonomy Behaviors and Algorithms* (E. Joa,
Vehicle Autonomy and Intelligence Lab, SNU, Fall 2026), Module 0 —
*Discretization, Rigid-Body Motion, and Vehicle Dynamics*. Every convention,
parameter and numerical claim below is traceable to that material, and the test
suite reproduces its figures.

> **한국어 요약** — 강의의 규약(ISO 좌표계, `α = 휠헤딩 − 속도방향`, `F_y = C_α α`)과
> 기준차량(`m=1500, I_z=2400, l_f=1.2, l_r=1.5, C_f=80k, C_r=100k`)을 그대로 사용하는
> 모듈형 2D 자율주행 시뮬레이터입니다. 적분기 수렴차수(1,2,2,4), SE(2) 상수 트위스트
> 드리프트(0.84 m), 언더스티어 구배(`K_us=3.75e-3`, `v_ch=26.8 m/s`), 피드포워드
> 분해표까지 강의 수치를 코드가 재현합니다. FWD 구동·제동·현가(하중이동)를 포함한
> 비선형 플랜트 위에서 인지 → 예측 → 행동결정 → 궤적생성 → NMPC의 전체 스택이
> 동작하며, 13개 시나리오에 대해 26개 KPI로 합격/불합격을 판정합니다.

---

## Install and run

```bash
pip install -e ".[dev,viz]"

avsim list                                   # the scenario suite
avsim run signal_red --plot red.png          # one scenario + a figure
avsim suite --out results/                   # everything, with reports and figures
avsim study                                  # the lecture's five study tasks
avsim contract                               # the model contract for one manoeuvre
pytest -q                                    # 80 tests
```

The core (`avsim.core`, `avsim.models`, `avsim.planning`, `avsim.control`)
depends on **numpy alone** — the matrix exponential, the cubic spline and the
optimizer are all implemented here. `matplotlib` is needed only for `avsim.viz`.

---

## Architecture

```
avsim/
  core/        conventions · integrators · SE(2) · geometry
  models/      params · tire · suspension · powertrain
               kinematic_bicycle · dynamic_bicycle · frenet · embedded · linearization
  world/       path · road · network · traffic_light · actors · world
  perception/  sensor · tracker
  planning/    prediction · velocity_profile · behavior · trajectory
  control/     geometric · ilqr · mpc
  autonomy/    stack
  eval/        kpi · scenarios · runner · studies
  viz/         render
```

The dependency arrow points one way: `world` knows nothing about `planning`,
and `planning` never reads ground truth. The ego stack receives other vehicles
only through a **sync source** and sees them only through the sensor, which is
the only way to be sure the planner is not cheating.

### One control tick

| step | module | in → out |
|---|---|---|
| 1 | `perception.sensor` | ground truth → detections (FOV, occlusion, noise, misses, latency) |
| 2 | `perception.tracker` | detections → confirmed tracks (CV Kalman, Mahalanobis gate) |
| 3 | `planning.prediction` | tracks → multi-modal predictions with anisotropic uncertainty |
| 4 | `planning.behavior` | predictions + signals → a decision **with a reason** |
| 5 | `planning.velocity_profile`, `planning.trajectory` | decision → reference trajectory |
| 6 | `control.mpc` | reference + constraints → `(a, delta)` |

The world integrates at 20 ms while the stack runs at 100 ms and its command is
**held** in between — the zero-order hold the whole modelling chapter is about.

---

## Conventions

Collected in `avsim/core/conventions.py`, because the lecture is blunt about it:

> Half the sign errors in this field come from mixing two conventions in one file.

* ISO: `x` forward, `y` **left**, `z` up; yaw `psi` counter-clockwise;
  steering `delta` positive **left**.
* Body origin at the **CG** unless a symbol is documented as rear-axle.
* Slip: `alpha = wheel heading − velocity direction`, so `F_y = C_alpha alpha`
  with `C_alpha > 0`.
* Two state vectors, never interchangeable term by term:
  `[X_r, Y_r, psi, v]` (rear axle) and `[X, Y, psi, v_x, v_y, r]` (CG).
  `kinematic_bicycle.rear_axle_to_cg` performs the lever-arm transform.

---

## What the models are, and what they are wrong about

| model | states | good for | omits |
|---|---|---|---|
| kinematic bicycle | 4 | urban planning, parking | tire forces, the friction limit |
| kinematic + understeer | 5 | the MPC prediction model | transients, saturation |
| common-state blend | 6 | manoeuvres spanning standstill | validated `tau`, `V*`, `dV` |
| dynamic bicycle (plant) | 13 | everything the ego is judged on | roll/pitch as *motion*, four-wheel effects |

The plant carries the rigid body (6), a roll/pitch **suspension** (4) and
**actuators** (3). The suspension never moves the body in X-Y: its only job is
to redistribute the four normal loads, which set the four friction budgets —
the honest way to keep a suspension in a planar model. Drive force is applied
at the **front (steered) axle only** (FWD), so the front contact patches carry
both the traction and the cornering demand, and the combined-slip ellipse binds
there first.

`DynamicBicycle.limit_balance()` reports what the reference parameters actually
imply: `K_us = +3.75e-3` is linear understeer, but the rear axle's linear reach
(3.37°) is shorter than the front's (5.27°), so the car drifts toward limit
oversteer above roughly 5 m/s². That is a property of the given parameters, and
it is reported rather than assumed.

---

## Reproduced from the lecture

Every row is asserted by the test suite.

| quantity | lecture | this code |
|---|---|---|
| observed integrator orders | 1, 2, 2, 4 | 1.00, 2.00, 2.00, 4.00 |
| Euler global error, 4 s at `h = 0.1` | ~0.7 m | 0.677 m |
| midpoint global error, same | ~2 mm | 2.10 mm |
| SE(2) group step, `v=10, ω=0.5, T=4` | 3e-14 m | 3.35e-14 m |
| Euler on the chart, same | 0.84 m | 0.8415 m |
| one step, chord vs arc | 2.5 cm | 2.50 cm |
| understeer gradient `K_us` | 3.75e-3 s²/m | 3.75e-3 |
| characteristic speed `v_ch` | 26.8 m/s (96 km/h) | 26.83 (96.6) |
| feedforward at 15 m/s, κ=0.01 | 0.0270 + 0.0084 = 2.03° | identical |
| feedforward at 30 m/s, κ=0.005 | 0.0135 + 0.0169 = 1.74° | identical |
| urban turn, 8 m/s on R=30 | `a_y = 2.13 m/s²` | 2.133 |
| Euler/RK4 unusable at `h=0.1` below | ~walking pace | 7 m/s (both) |

The steering feedforward is measured, not assumed. On a 200 m radius at 25 m/s:

| controller | kinematic plant | dynamic plant |
|---|---|---|
| pure pursuit | 0.00 m | −1.17 m |
| pure pursuit + ff | +0.76 m | −0.38 m |
| Stanley | +0.02 m | −0.46 m |
| Stanley + ff | +0.35 m | −0.13 m |

On the tireless model the feedforward *hurts*; on the vehicle that has tires it
removes two thirds of the error. A feedforward is a statement about a model.

---

## The optimal control problem

```
min  Σ q(x_k, u_k) + p(x_N)
s.t. x_{k+1} = F_h(x_k, u_k)
     (x_k, u_k) ∈ C
```

* **State** `[X, Y, psi, v, delta]` at the rear axle; **input** `[a, delta_dot]`.
  Making the steering *rate* the input turns the lecture's rate constraint
  `|delta_k − delta_{k−1}| ≤ delta_dot_max h` into a box the backward pass
  enforces exactly and for free.
* **Discretization** RK4, with Jacobians taken **through the RK4 step**
  (`linearization.rk4_jacobians`) — derivatives of the constraint the solver
  actually evaluates, 7× faster than finite differences.
* **Prediction model** `psi' = v tan(delta) / (L + K_us v²)`, which reproduces
  `delta_ss = (L + K_us V²) κ` exactly. Without it the model has no understeer,
  the optimizer commands the geometric angle and the car runs wide.
* **Cost** in the **path frame** (`[e_lon, e_lat, e_psi, e_v, delta]`), so the
  lateral gain does not depend on which way the road points.
* **Constraints** steering, speed, lateral-acceleration budget, the lane
  corridor as two affine half-spaces per node, obstacle avoidance as oriented
  ellipses between multi-circle body covers, and the stop line as one half-space
  on the front bumper. All with analytic Jacobians, verified to 1e-9.
* **Solver** augmented-Lagrangian iLQR with a control-limited backward pass. It
  matches the analytic finite-horizon LQR to 4e-16 on an unconstrained problem.
  Warm-starting **multipliers as well as inputs** takes a receding-horizon step
  from 112 ms to 21 ms; a wall-clock budget bounds it and is reported in the
  status rather than hidden.

Transcription is single shooting, so the dynamics are feasible by construction
and only path constraints can be violated — the trade the lecture names, chosen
because the horizons are short and no QP is then needed.

---

## Inter-sample feasibility

The lecture is explicit that constraints hold **at the nodes, only there**, and
this code does not pretend otherwise. Two tightenings are applied and both are
labelled as such:

* obstacles are grown by half the distance they travel between samples;
* the lane corridor is tightened by the vehicle half-width.

Neither is a proof. `avsim contract` prints this as part of the model contract.

---

## Scenarios and KPIs

13 scenarios, from lane keeping to the lecture's running example (an unprotected
left across oncoming traffic). Reactive traffic (`SimulatedTrafficSource`, IDM +
pure pursuit + signal compliance) asks whether the plan survives traffic that
responds to it; scripted traffic (`ScriptedSyncSource`) reproduces one
adversarial event identically on every run. Both are needed and they answer
different questions.

26 KPIs in six groups — **safety** (collisions, clearance, TTC, red-light
compliance, corridor departure, friction usage, rollover margin), **tracking**,
**comfort**, **progress**, **compute** (solve time, real-time factor, solver
success, fallback use) and **perception**. Each carries a threshold and a
direction, so a report says PASS or FAIL rather than leaving the reader to
decide. Perception recall is measured against what the sensor could **actually
see**, so an occluded vehicle is not scored as a tracker failure.

---

## Current status

`avsim suite` on this code, one clean run:

| scenario | verdict | note |
|---|---|---|
| `lane_keeping` | PASS | 0.00 m settled cross-track at 16 m/s |
| `curved_lane_keeping` | PASS | 0.06 m RMS on a 300 m radius |
| `static_obstacle` | FAIL | hairline: 2.518 m against a 2.5 m bound on a *lane-change* RMS |
| `blocked_single_lane` | PASS | stops 4.3 m short with no room to pass |
| `lead_braking` | FAIL | hairline: 8.008 m/s² against an 8 m/s² bound |
| `cut_in` | PASS | |
| `low_mu_curve` | FAIL | hairline: 5.084 m/s² against a 5 m/s² bound |
| `signal_green` / `signal_red` / `signal_yellow_dilemma` | PASS | including the dilemma-zone decision |
| `tight_right_turn` | FAIL | **open**: see below |
| `unprotected_left` | PASS | the lecture's running example, completed |
| `cross_traffic` | PASS | yields to a red-light runner |

Two things this table is honestly saying.

**`tight_right_turn` is an open defect.** It passes when run alone and fails
inside the suite. The cause is the MPC's wall-clock budget: under load the
solver gets fewer iterations, the lattice then rejects more candidates on the
5.75 m radius, and the fallback takes over (91% of ticks). The budget is kept
because without it the solve tail reaches 400 ms and the loop is not real time;
the trade is documented at `MPCConfig.time_budget` and `time_budget=None` makes
runs reproducible at that cost. The right fix is a cheaper standstill and
low-speed regime, not a longer budget.

**Three hairline failures are left failing on purpose.** 2.518 against 2.5 and
8.008 against 8 could be made green by moving the bound. They are the KPI layer
doing its job, and moving a threshold to cover a number it was written to catch
is how a test suite stops meaning anything.

---

## Known limitations

Stated rather than discovered:

* **Perception** models FOV, occlusion, range-dependent noise, misses and
  latency — but no false positives, no classification error, no calibration
  error. A KPI measured here is optimistic about perception.
* **Prediction** is constant-velocity per mode. It is systematically wrong for a
  braking or turning vehicle; the multi-modal lane predictor covers intersection
  turns but not lane changes.
* **Suspension** is roll/pitch for load transfer only. There is no pitch-induced
  change of the *path*, no aerodynamic load, no per-wheel steering geometry.
* **Tires** use a load-scaled Magic Formula with no relaxation length in the
  force path, no camber, no temperature and no combined-slip longitudinal
  saturation beyond the friction ellipse.
* **The solver** is single shooting with a wall-clock budget. When the budget
  binds the returned solution is suboptimal and possibly constraint-violating;
  the stack falls back to the geometric controller and **records that it did**.
* **Inter-sample feasibility** is tightened, not guaranteed (above).
* **Runs are not bit-reproducible** while the MPC has a wall-clock budget,
  because the number of iterations depends on machine load. Set
  `MPCConfig.time_budget = None` for reproducibility and accept a longer
  solve-time tail.
* **The low-speed regime is the weak one**, exactly where the lecture says it
  will be. At rest the prediction model loses steering authority, the augmented
  Lagrangian fights the `v >= 0` bound, and the stack has to hold the steering
  and skip the solve to stay real time. That is a workaround, not a model.

---

## References

* E. Joa, *Discretization, Rigid-Body Motion, and Vehicle Dynamics*, Module 0,
  Introduction to Autonomy Behaviors, SNU, Fall 2026.
* Borrelli, Bemporad & Morari, *Predictive Control for Linear and Hybrid
  Systems*, Ch. 2–3.
* Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Ch. 1–2 and 13.
* Bock & Plitt (1984), multiple shooting.
* Werling et al. (2010), optimal trajectories in a Frenet frame.
* Tassa, Mansard & Todorov (2014), control-limited differential dynamic
  programming.
* Treiber, Hennecke & Helbing (2000), the Intelligent Driver Model.
