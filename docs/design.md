# Design notes

Why the code is shaped the way it is. Each section names a decision, the
alternative, and what the alternative costs.

---

## 1. One conventions module, imported everywhere

`avsim/core/conventions.py` fixes the ISO frame, the slip-angle sign, the two
state layouts and the angle-wrapping branch cut. Nothing else defines them.

*Alternative*: state the convention in each module's docstring.
*Cost*: the lecture's own warning — half the sign errors in this field come from
mixing two conventions in one file. A shared `wrap_to_pi` in particular means
the simulator, the estimator and the optimizer cut the circle in the same place;
when they do not, a controller commands full lock for one step as the vehicle
crosses ±π and nothing in the logs explains it.

---

## 2. Integrators as Butcher tableaux, with stability functions

Euler, midpoint, Heun and RK4 share one `step`, and each carries its stability
polynomial `R(z)`.

*Alternative*: hand-write each method.
*Cost*: the stability function is what turns "which speeds can I integrate at
`h = 0.1 s`?" into `max_stable_step`, a bisection on `|R(hλ)| ≤ 1`. Without it,
Study Task 3 is a guess. It also makes the low-speed failure legible: the linear
single-track model is unintegrable below ~7 m/s at that step for *both* methods,
which says the model must change, not the step.

---

## 3. The plant is 13-dimensional; the prediction model is 5

The plant carries the rigid body (6), a roll/pitch suspension (4) and actuators
(3). The MPC predicts with `[X, Y, psi, v, delta]`.

*Alternative*: predict with the plant.
*Cost*: solve time, and no benefit — the lecture's rule is that more fidelity
helps only when the error it removes is worth the state dimension, the
nonlinearity and the solve time it adds. Instead the mismatch is **measured**:
`max_friction_usage` and `max_lat_accel` are KPIs, so a plan the tires refuse
shows up as a number rather than as an unexplained tracking failure.

---

## 4. Suspension exists, but never moves the body

Roll and pitch are two damped oscillators driven by the body accelerations.
Their only output is the four normal loads.

*Alternative*: omit the suspension, or make it move the vehicle in 3-D.
*Cost*: omitting it makes every tire carry a static load, so the friction budget
is wrong exactly when it binds — under combined braking and cornering. Making it
3-D leaves the stated scope. The chosen middle keeps the motion planar and the
budgets honest, and the gravity term `−m g h_roll` is kept explicit so a badly
chosen roll stiffness appears as an unstable roll mode instead of quietly
optimistic loads.

---

## 5. Front-wheel drive is a modelling commitment, not a label

Traction is applied at the front axle only, and the brakes are split by a fixed
bias. The combined-slip ellipse is then evaluated per axle.

*Cost of ignoring it*: the front contact patches carry the traction *and* the
cornering demand, so an FWD car loses cornering capacity exactly when the
throttle is applied mid-corner. A model that spreads drive force evenly cannot
represent that, and a planner tuned against it will be wrong at the limit in the
direction that matters.

---

## 6. The steering *rate* is the MPC input

State `[X, Y, psi, v, delta]`, input `[a, delta_dot]`.

*Alternative*: input `[a, delta]` with a rate constraint
`|delta_k − delta_{k−1}| ≤ delta_dot_max h`.
*Cost*: that constraint couples consecutive stages, which an augmented
Lagrangian must then carry as an extra multiplier per stage. As a *box on the
input* the backward pass enforces it exactly, for free, and the commanded
steering is continuous by construction — so the plant's rate-limited actuator is
never asked for a slew it cannot perform.

---

## 7. The prediction model has understeer

`psi' = v tan(delta) / (L + K_us v²)` rather than `v tan(delta) / L`.

*Alternative*: the plain kinematic bicycle.
*Cost*: measured. On a 300 m radius at 16 m/s the plain model commands the
geometric angle, the real vehicle runs wide, and the feedback is left to
generate a steady-state demand the model already knows about — 1.9 m of drift
where the corrected model holds the lane. This is the lecture's "feedforward
handles the path; feedback handles the error", moved inside the model.

---

## 8. The MPC cost is written in the path frame

Residual `[e_lon, e_lat, e_psi, e_v, delta]`, with the position error rotated
into the reference tangent and normal.

*Alternative*: weight `X` and `Y` directly.
*Cost*: the lateral gain would depend on which way the road points — 6 on a road
running north, 2 on one running east, and anything between on a curve. The bug
this caused was not a crash; it was a controller that tracked well on one
heading and drifted on another.

---

## 9. A lattice *and* an MPC

The Frenet lattice samples discrete manoeuvres; the MPC tracks the winner under
constraints.

*Alternative*: MPC alone.
*Cost*: the lecture is explicit — linearizing an obstacle gives one affine
half-space, the nominal trajectory chooses which side to pass, and the convex
subproblem never explores the other. A gradient method cannot switch homotopy
class. The lattice is that search; the MPC is what makes the chosen class
executable.

---

## 10. Re-plan from the nominal, not from the measurement

The lattice is seeded with the previous plan's state one control step in, and
re-seeded from the measurement only when the two diverge by more than 0.6 m.

*Alternative*: seed from the measurement every tick.
*Cost*: measured, and it is the largest single effect in this codebase. Every
plan promises to return to the centerline over its horizon; only the first
0.1 s of it is executed; the next plan makes the same promise from a slightly
worse state. On a 300 m radius at 16 m/s that is a 1.6 m excursion where
planning from the nominal gives 0.13 m — the same figure the MPC achieves
tracking the route directly, which is the check that isolated the cause.

---

## 11. Multi-circle bodies and anisotropic uncertainty

Vehicles are covered by three circles (r = 1.20 m) rather than one (2.48 m), and
prediction uncertainty has separate longitudinal and lateral axes.

*Alternative*: one enclosing circle and one sigma.
*Cost*: a single circle is 2.7× the car's half-width, so every lane change reads
as a collision. An isotropic sigma inherits the longitudinal growth — which is
large, because an unknown acceleration integrates twice — into the lateral
direction, and forbids passes that are wide open. Together these two
approximations turn a routine overtake into a hard stop.

---

## 12. The behaviour layer returns constraints, not commands

Every decision carries a `reason` string, and the KPI layer logs it.

*Cost of the alternative*: "why did it brake?" becomes unanswerable. The
timeline plot and the behaviour column of the telemetry exist for exactly this.

---

## 13. Warm-start the multipliers, and bound the solve

The MPC carries inputs, multipliers and the penalty across steps, and stops at a
wall-clock budget with an honest status.

*Alternative*: restart the augmented Lagrangian each step, or run to
convergence.
*Cost*: restarting makes the receding horizon cost *more* per step than a cold
solve (112 ms against 21 ms), because the solver rediscovers which constraints
are active. Running to convergence overruns the control period, at which point
the controller has solved a different problem, late. The budget is reported in
`solve_time_p95` and `real_time_factor`, and every fallback is counted.

---

## 14. Two kinds of longitudinal bound

`stop_s` is a hard stop line (red light, obstacle). `safety_bound_s` is a soft
one (a following distance).

*Cost of conflating them*: "the ego must be *able* to stop before X" is a speed
ceiling `v ≤ sqrt(2b(X − s))`, not an instruction to stop at X. Treating an RSS
following distance as a stop line makes the vehicle brake to rest fifty metres
behind a car still doing 14 m/s.

---

## 15. Diagnostics are refreshed at the accepted state

After each integration step the plant's derivative is evaluated once more at the
state that was accepted.

*Cost of the alternative*: during an RK4 step the diagnostics are last written by
an internal stage evaluated at `x + h k3` — a point that is not on the
trajectory and, near standstill under hard braking, not physical. Reading them
reported 8.6 m/s² of lateral acceleration on a vehicle standing still, and every
comfort KPI built on it was wrong.

---

## 16. The tracker's `dt` must equal its update interval

Constructing a Kalman filter with `dt = 0.05` and calling `update()` every
0.1 s propagates half the motion each step. The velocity estimate then lags a
decelerating leader by seconds — which was a rear-end collision, not a
cosmetic error. The stack now raises if the two disagree.

---

## 17. A wall-clock budget buys real time and sells reproducibility

The MPC stops at 45 ms. That is what a vehicle has, and without it the solve
tail near standstill reaches 400 ms — the loop is then not real time at all.

The cost is that the number of iterations depends on machine load, so the same
scenario can converge alone and be cut short inside a suite. `tight_right_turn`
does exactly that, and it is recorded as an open defect rather than hidden by a
longer budget.

The reduction that worked was removing the *pathology* rather than raising the
cap: while holding a stop, the stack skips the solve entirely, because the
prediction model is degenerate there and the answer is already known. That
dropped the mean solve from 184 ms to 20-75 ms and the real-time factor from
1.85 to 0.2-0.8. What remains is the low-speed regime itself, which needs a
low-speed-valid model — the lecture's own conclusion — not more compute.

---

## 18. The platform runs the solver to its iteration bound, not to a clock

Decision 17's trade is right on a vehicle and wrong on a bench. The platform's
job is to *compare* algorithms, and a solver cut short by machine load makes one
configuration produce two different answers — the one thing a comparison cannot
survive. So `avsim.platform.controllers.MPCController` sets
`time_budget = None`: the work is bounded by iterations, and the compute time is
**measured and scored** (`real_time_factor`, its own KPI category) rather than
enforced.

Two runs of the same configuration were checked to agree exactly on
`(time_to_goal, red_light_violations, distance)`. The price was a mean solve of
140 ms against a 100 ms control period, which decision 19 paid back.

---

## 19. Projection goes through the cached table

`ReferencePath.project` did its coarse scan as a Python loop over
`position(s)` — 200 scalar segment lookups per call, on a path that already
keeps an interpolation table for exactly this reason. One planning tick projects
tens of points (the MPC reference alone is 26), so the loop was **half of all
simulation time**: it dominated the optimizer it exists to feed.

The scan now goes through `frames()`, which is three `np.interp` calls on the
cached table. Accuracy is unchanged — the scan only chooses which basin the
Newton iteration starts in, and the table's spacing *is* the scan's spacing;
projection error against known points is zero to machine precision. A 20 s run
went from 55 s to 24 s of compute and the MPC's real-time factor from 1.40 to
0.73, with the closed-loop trajectory bit-identical.

The general lesson is the one the lecture makes about model choice: the
expensive thing was not the algorithm anybody was thinking about.

---

## 20. Other vehicles obey the road, not the ego

The grid traffic in `avsim.world.grid_traffic` is not autonomous and does not
cooperate: it follows IDM behind its own leader, obeys its own signals, reserves
the intersection box, and yields to crossing traffic that got there first. It
does not know the ego exists beyond treating it as one more obstacle.

The one guarantee is that traffic vehicles never drive into *each other*.
Without it a collision would be ambiguous — the platform could not say whether
the ego caused it — and a safety KPI computed from an ambiguous collision is
worse than no KPI. It is asserted box-to-box across a 90 s run in
`tests/test_platform.py`, not assumed from the car-following model.

---

## 21. Weights are a reporting choice, so re-scoring never re-runs

Mission time, safety margin and energy have different dimensions, and no
exchange rate between them is a property of the simulation. Fixing one in code
would be the platform asserting an answer to the question it exists to let the
user ask.

So the session stores **raw metrics**, and `ScoreConfig` — every threshold,
every weight, the collision policy — is applied afterwards. `/api/rescore`
re-scores every stored run instantly. "What if safety mattered three times as
much?" has to be cheap, or it will not get asked.

The same reasoning made metric scoring direction-aware at infinity:
`min_clearance = inf` means nothing was near and scores 100, while
`time_to_goal = inf` means the mission never finished and scores 0. A bare
finiteness check punished an empty road, which is a scoring bug that looks like
a driving result.

---

## 22. The controller contract hides ground truth, on purpose

A plug-in receives `Observation`: the ego's own state, **tracked** objects with
their covariance trace, the route ahead, the next signal, and the vehicle it is
driving. It does not receive the true state of other vehicles. Detections come
from the sensor and are smoothed by the tracker, so they can be missing, late,
or duplicated, and track ids are not stable.

A controller that needs ground truth cannot be evaluated on this platform, which
is the point. The escape hatch (`obs.extras`) exists because the built-in stack
is part of the package and needs the road network, and it is documented as not
being part of the contract.

Commands are normalized — `steer` in `[-1, 1]` scaled by `delta_max`, `throttle`
and `brake` in `[0, 1]` scaled by the limits — so the same controller stays
correct when the parameter sliders change the car underneath it. Brake wins over
throttle rather than being averaged, and a non-finite command raises instead of
travelling into the plant to reappear as a vehicle at infinity ten seconds later.

---

## 23. A stop line is a nose limit; the plan is a rear-axle arc length

`BehaviorDecision.stop_s` is where the **front bumper** must not pass, because
that is what a stop line is and what the MPC's half-space constrains. Every
planning arc length, though, is measured at the **rear axle** — the kinematic
bicycle's reference point. The two differ by the front overhang, 3.7 m on the
reference car.

Handing the same number to both is a bug in one direction or the other: give
the nose limit to the velocity profile and the car parks with its nose 1.7 m
across the line (a violation in every stopping scenario, in every controller);
subtract the overhang twice and it stops a car-length short of a line it was
asked to reach. The stack converts once, in `_rear_axle_stop`, and the
conversion is named rather than inlined so it cannot be applied twice.

---

## 24. A commanded stop is not scored against the cruise target

The lattice samples a *stop* candidate (a quintic ending at the stop point with
zero speed) alongside *keep* candidates aimed at a target speed. Scoring them
with the same speed cost charges the stop candidate `w_speed * mean(v - v_target)^2`
— about 150 points at 8 m/s — for doing exactly what it was asked. It then
loses to a candidate that merely slows down, and the vehicle arrives at the line
still moving and crosses it. The stop existed in the candidate set on every
tick of the approach and never won once.

So a stop candidate carries no speed cost: its speed profile is determined by
the stop point, which is the intent. The speed target is the intent only when
there is no stop. Everything else about the candidate — jerk, offset,
consistency, risk, and the feasibility filter — still applies, so a stop that
cannot be made comfortably from here is still rejected, and the keep candidates
carry the approach until it can.

This is the same lesson as decision 14, one level up: two different intents need
two different costs, and merging them silently makes one of them unreachable.

---

## 25. Near standstill the lateral polynomial is taken over distance

Werling's lattice plans `e_y(t)` as a minimum-jerk polynomial in **time**. At
speed that is right: the vehicle covers metres while it moves metres sideways,
and the implied path curvature is small.

From rest it is wrong, and the failure is total. A polynomial in time puts a
1.5 m correction into the first second whatever the vehicle is doing; from rest
that first second covers about a metre of road, so the implied curvature is
enormous and every candidate is rejected — for curvature, for lateral
acceleration, or for leaving the corridor. The vehicle is then trapped: it
cannot pull away, so it never reaches the speed that would make pulling away
plannable. In an unprotected left it sat 1 m off the centreline for the last
34 s of the run.

Below 3 m/s the polynomial is therefore taken over the distance the candidate
travels — Werling's own low-speed formulation. The correction is spread over
metres of road instead of seconds of clock and is feasible by construction:
1.75 m returned over 12 m of travel is a 0.07 1/m curve against a 0.25 limit.
The return distance is bounded rather than being the candidate's whole travel,
because a 4.5 s horizon from rest covers 30 m and spreading the correction over
all of it leaves the vehicle wide for the entire manoeuvre.

---

## 26. "Holding a stop" is a claim about the horizon, not about the next tick

The stack skips the MPC solve while it is stopped and means to stay stopped —
decision 17's cheap standstill. The test was the plan's speed 0.6 s ahead.

A minimum-jerk pull-away is still under 0.3 m/s after 0.6 s. So the test read
"stopped" for every departure, the solve was skipped, the commanded
acceleration was the plan's first sample — zero, for a curve that starts flat —
and the vehicle never moved again. The lattice was returning plans and the MPC
was converging the whole time; nothing looked broken except that the car was
stationary.

The test is now the plan's *maximum* speed over its horizon, which is what
"intends to stay stopped" actually means. The lesson is narrow and worth
keeping: a predicate about intent must be evaluated over the interval the
intent covers.
