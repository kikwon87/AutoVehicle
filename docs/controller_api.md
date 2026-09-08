# Control Algorithm Interface Report
# 제어 알고리즘 인터페이스 규격서

**Contract version 1.0** — `avsim.platform.controller_api.CONTRACT_VERSION`

This document is the whole agreement between the test platform and a control
algorithm. If you want to write your own controller — MPC, a rule set, a neural
network, anything — everything you need is here: what you are given, what you
must return, how your file is found and loaded, and how the result is scored.

이 문서는 테스트 플랫폼과 제어 알고리즘 사이의 계약 전부입니다. 직접 제어기를
작성하려면(MPC, 규칙 기반, 신경망 무엇이든) 필요한 내용은 모두 여기에 있습니다:
무엇을 입력으로 받는지, 무엇을 반환해야 하는지, 파일이 어떻게 로드되는지,
결과가 어떻게 채점되는지.

---

## 1. The contract in one sentence / 한 문장 요약

> A controller is an object with a `control(obs) -> command` method. Once per
> control tick (default **0.1 s**) the platform calls it with an `Observation`
> and expects a `ControlCommand` back.

> 제어기는 `control(obs) -> command` 메서드를 가진 객체입니다. 제어 주기마다
> (기본 **0.1초**) 플랫폼이 `Observation`을 넘겨 호출하고 `ControlCommand`를
> 돌려받습니다.

Minimal working plug-in — save as `my_controller.py` and select it in the UI:

```python
class MyController:
    name = "My first controller"
    description = "Proportional lane keeping, constant speed."

    def reset(self, context):
        pass                                   # clear per-run state here

    def control(self, obs):
        target = obs.route_at(8.0)             # a point 8 m ahead on the route
        heading_error = obs.e_psi              # radians, + = pointing left
        steer = -1.2 * obs.e_y - 2.0 * heading_error
        steer /= obs.vehicle.delta_max         # normalize to [-1, 1]

        want = min(obs.speed_limit, 10.0)
        err = want - obs.ego.v
        return {"steer": steer,
                "throttle": max(0.0, 0.4 * err),
                "brake": max(0.0, -0.4 * err)}
```

That is a complete, loadable controller. No import from `avsim` is required —
the platform duck-types, and a plain `dict` is accepted in place of a
`ControlCommand`.

---

## 2. Conventions / 좌표 및 부호 규약

Everything follows ISO 8855, the same convention as the lecture material and
the rest of the package (`avsim.core.conventions`).

| Quantity | Symbol | Unit | Sign |
|---|---|---|---|
| longitudinal / forward | `x` | m | forward positive |
| lateral | `y` | m | **left** positive |
| yaw / heading | `psi` | rad | counter-clockwise positive |
| steering (road wheel) | `delta` | rad | **left turn positive** |
| lateral path offset | `e_y` | m | vehicle left of the path is positive |
| curvature | `kappa` | 1/m | left-hand bend positive |

Angles are radians everywhere, never degrees. Time is seconds, distance metres,
mass kilograms. `psi` is wrapped to `(-pi, pi]`.

Pose reference point: `ego.x`, `ego.y`, `ego.psi` are the **rear axle**, because
that is the reference of the kinematic bicycle model used throughout. The
velocities `v_x`, `v_y`, `yaw_rate`, `a_x`, `a_y` are body-frame quantities at
the **CG**, which is what an IMU measures.

자세 기준점은 **후륜축**이고, 속도·가속도·요레이트는 **무게중심(CG)** 기준의
차체 좌표계 값입니다. 각도는 모두 라디안입니다.

---

## 3. Inputs — the `Observation` / 입력

One immutable `Observation` per tick. Its fields:

### 3.1 Top level

| Field | Type | Meaning |
|---|---|---|
| `t` | float | simulation time [s] since the run began |
| `dt` | float | control period [s]; the interval until the next call |
| `ego` | `EgoState` | own state, see §3.2 |
| `objects` | tuple of `DetectedObject` | perceived vehicles, see §3.3 |
| `route` | tuple of `RoutePoint` | the reference path ahead, see §3.4 |
| `s` | float | ego arc length along the route [m] |
| `e_y` | float | lateral offset from the route centreline [m], + = left |
| `e_psi` | float | heading error relative to the route tangent [rad] |
| `lane_bounds` | (float, float) | allowed lateral corridor `(min, max)` [m] |
| `speed_limit` | float | posted limit on this route [m/s] |
| `goal_s` | float | arc length at which the mission is complete [m] |
| `signal` | `SignalInfo` or `None` | next signalized stop line, see §3.5 |
| `vehicle` | `VehicleInfo` | parameters of the car you are driving, see §3.6 |
| `extras` | dict | free-form; **nothing in the contract depends on it** |

`extras` carries the platform's internal objects (the ego actor, the raw actor
list, the tracker output). The built-in MPC uses them because it is part of the
package. A plug-in should not: they are not stable API, and a controller that
reads ground truth from them is not being evaluated on perception.

### 3.2 `obs.ego` — `EgoState`

| Field | Unit | Meaning |
|---|---|---|
| `x`, `y` | m | rear-axle position in the world frame |
| `psi` | rad | heading |
| `v` | m/s | speed at the rear axle |
| `v_x`, `v_y` | m/s | body-frame velocity at the CG |
| `yaw_rate` | rad/s | yaw rate |
| `delta` | rad | **realized** road-wheel angle (the actuator's state, not your command) |
| `a_x`, `a_y` | m/s² | body-frame accelerations |
| `beta` | rad | dynamic sideslip, `atan2(v_y, v_x)` |

`delta` is what the wheels are actually at. The steering actuator is a
first-order lag with a rate limit, so it trails your command; that gap is real
and is yours to handle.

### 3.3 `obs.objects` — `DetectedObject`

Other vehicles as **the perception stack estimates them**, not as they are.
Detections come from a simple vision model (limited range, limited field of
view, occlusion, noise) and are smoothed by a constant-velocity Kalman tracker.

| Field | Unit | Meaning |
|---|---|---|
| `id` | int | **track** id — not a ground-truth id |
| `x`, `y` | m | estimated position (vehicle centre) |
| `psi` | rad | estimated heading |
| `v` | m/s | estimated speed |
| `v_x`, `v_y` | m/s | estimated world-frame velocity |
| `length`, `width` | m | estimated box size |
| `range`, `bearing` | m, rad | relative to the ego, for convenience |
| `age` | int | consecutive updates this track has survived |
| `position_variance` | m² | trace of the position covariance — how much to trust it |

Three consequences worth stating plainly:

* Track ids **are not stable**. A lost-and-reacquired vehicle gets a new id, and
  two vehicles passing close by can swap. A controller that keys behaviour on id
  identity will be surprised.
* An object you cannot see is **absent from the list**, not marked as unknown.
  Occlusion is real, especially at intersections.
* A young track (`age` small, `position_variance` large) has a velocity estimate
  that is mostly noise. Wait for it, or weight it down.

인지 결과는 **추정치**입니다. 트랙 ID는 유지된다는 보장이 없고, 가려진 차량은
목록에서 아예 빠지며, 갓 생성된 트랙의 속도는 대부분 잡음입니다.

### 3.4 `obs.route` — `RoutePoint`

Samples of the reference path, from the ego forward, by default 120 m at 4 m
spacing (`RunConfig.route_horizon`, `route_sample_ds`). Fields: `s`, `x`, `y`,
`heading`, `curvature`, `speed_limit`. The list is ordered by increasing `s` and
is clipped at the end of the route, so near the goal it stops advancing.

Helper: `obs.route_at(distance)` returns the sample `distance` metres ahead of
the ego — the usual first ingredient of a pure-pursuit or Stanley controller.

### 3.5 `obs.signal` — `SignalInfo`

`None` when no signalized stop line lies ahead on the route. Otherwise:

| Field | Meaning |
|---|---|
| `group` | signal group id, e.g. `"n1_1:NS"` |
| `colour` | `"green"`, `"yellow"` or `"red"` |
| `distance` | arc length from the ego to the stop line [m]; goes negative once past |
| `time_to_change` | seconds until this signal next changes colour |

A violation is recorded when the **front bumper** crosses the stop line while the
signal is red. `time_to_change` is exact, not an estimate — the platform gives
you the same information a well-instrumented V2I link would, so that dilemma-zone
decisions are a control problem rather than a guessing game.

정지선 통과 판정은 **앞 범퍼** 기준입니다. 차량 기준점(후륜축)이 아닙니다 —
`obs.vehicle.length`와 축거로 오버행을 계산해 두는 편이 안전합니다.

### 3.6 `obs.vehicle` — `VehicleInfo`

The car you are driving *right now*, after the user's parameter sliders were
applied. Read it rather than hard-coding numbers, and the same controller stays
correct when the vehicle changes underneath it.

| Field | Unit | Meaning |
|---|---|---|
| `wheelbase`, `l_f`, `l_r` | m | axle geometry |
| `mass`, `yaw_inertia` | kg, kg·m² | inertial properties |
| `length`, `width` | m | body box |
| `delta_max` | rad | steering limit — the scale of your `steer` output |
| `delta_rate_max` | rad/s | steering rate limit of the actuator |
| `a_max`, `a_min` | m/s² | accel limit (positive) and brake limit (**negative**) |
| `mu` | – | tire–road friction the platform is configured with |
| `understeer_gradient` | s²/m | `K_us`; steady-state steer is `(L + K_us V²)·kappa` |

`mu` is the configured value; a low-friction patch on the road may be worse. The
friction envelope is `sqrt(a_x² + a_y²) <= mu·g`, and how close you come to it is
a scored metric.

### 3.7 `reset(context)` — `ScenarioContext`

Called once before every run, before the first `control()`. Clear all per-run
state here — the same controller object is reused across runs of a batch.

| Field | Meaning |
|---|---|
| `scenario` | preset key, e.g. `"unprotected_left"` |
| `vehicle` | the same `VehicleInfo` as in the observation |
| `control_dt` | control period [s] |
| `goal_s` | mission-complete arc length [m] |
| `route_length` | total route length [m] |
| `speed_limit` | posted limit [m/s] |
| `seed` | the run's random seed — use it if you randomize anything |
| `options` | dict of options the user typed in the UI, passed straight through |

---

## 4. Output — the `ControlCommand` / 출력

Three normalized numbers. Nothing else reaches the vehicle.

| Field | Range | Physical meaning |
|---|---|---|
| `steer` | `[-1, 1]` | `delta_cmd = steer * vehicle.delta_max`; **+1 is full left** |
| `throttle` | `[0, 1]` | `a_cmd = throttle * vehicle.a_max` |
| `brake` | `[0, 1]` | `a_cmd = brake * vehicle.a_min` (`a_min < 0`) |
| `info` | dict | free-form; written to the run log beside the command |

Rules:

* **Brake wins.** If `brake > 0` the throttle is ignored — no averaging, no
  error. That is what a real brake-override does, and a controller leaking
  throttle during a stop should be visible in the log rather than smoothed away.
* Values outside range are **clipped**, not rejected.
* `NaN` or infinity is **rejected** with an exception that names your controller.
  A non-finite command propagates silently into the plant and reappears ten
  seconds later as a vehicle at infinity; failing loudly is cheaper.
* `steer` is a **position** command, not a rate. The actuator applies its own
  first-order lag and rate limit, so a discontinuous command is not a crash —
  but it is scored as steering effort.

브레이크가 우선입니다. 범위를 벗어난 값은 잘라내고, `NaN`은 예외로 거부합니다.
`steer`는 각도 지령이며(속도 지령이 아님) 액추에이터의 지연과 속도 제한을 거쳐
실제 조향각이 됩니다.

### Accepted return types / 허용되는 반환 형식

All four of these are accepted, so a first attempt is never blocked by a type:

```python
return ControlCommand(steer=0.1, throttle=0.3)      # the dataclass
return {"steer": 0.1, "throttle": 0.3}              # a dict
return (0.1, 0.3, 0.0)                              # (steer, throttle, brake)
return (0.1, 0.3)                                   # (steer, accel): + throttle, - brake
```

### Thinking in physical units / 물리 단위로 계산하는 경우

If your algorithm produces `a [m/s²]` and `delta [rad]` directly — as an MPC
does — convert with the helper rather than by hand:

```python
from avsim.platform.controller_api import ControlCommand
return ControlCommand.from_physical(a, delta, obs.vehicle, cost=J, iters=n)
```

Keyword arguments become `info`, which lands in the log. The inverse is
`cmd.to_physical(obs.vehicle) -> (a, delta)`.

---

## 5. How your file is loaded / 플러그인 로딩 규칙

Point the platform at a `.py` file (UI field "Controller file", or
`RunConfig.controller = "/path/to/my_controller.py"`). The loader looks for a
controller in this order:

1. `create_controller(**options)` — a factory function (**preferred**: it
   receives the options the user typed in the UI);
2. a class literally named `Controller`;
3. the single subclass of `avsim.platform.controller_api.Controller` in the file;
4. an already-built object named `controller`.

Two subclasses with no factory is an ambiguity the loader refuses to resolve on
your behalf — add a `create_controller()` and say which one you mean.

The object must have `control(obs)`. `reset(context)` and `diagnostics()` are
optional. `name` and `description` are shown in the UI and written into the
results table.

> **Loading a plug-in executes the file.** That is the intended behaviour — the
> platform exists to run algorithms people write — but it means a plug-in is
> exactly as trustworthy as its author. Read a file before you load it, the same
> as any other code you run.
>
> 플러그인 로딩은 해당 파일을 **실행**합니다. 의도된 동작이지만, 신뢰할 수 있는
> 코드만 로드하십시오.

### `diagnostics()` — optional

Called right after `control()`; return a flat dict of numbers or strings. They
are stored per tick and are the cheapest way to see, after the fact, why your
controller did what it did. The built-in MPC reports its cost, constraint
violation, solver status and solve time this way.

---

## 6. An ML controller / 학습 기반 제어기

Nothing in the contract mentions optimization or models, so a learned policy is
a plug-in like any other. The pattern:

```python
import numpy as np

class PolicyController:
    name = "Behaviour-cloning policy"

    def __init__(self, weights="policy.npz"):
        z = np.load(weights)
        self.W, self.b = z["W"], z["b"]

    def features(self, obs):
        lead = obs.lead_object()
        gap = lead.range if lead else 100.0
        rel_v = (lead.v - obs.ego.v) if lead else 0.0
        return np.array([
            obs.e_y, obs.e_psi, obs.ego.v / max(obs.speed_limit, 1e-3),
            obs.route_at(10.0).curvature, obs.ego.delta / obs.vehicle.delta_max,
            min(gap, 100.0) / 100.0, np.clip(rel_v / 10.0, -1.0, 1.0),
        ])

    def reset(self, context):
        pass

    def control(self, obs):
        y = np.tanh(self.W @ self.features(obs) + self.b)
        return {"steer": float(y[0]),
                "throttle": float(max(y[1], 0.0)),
                "brake": float(max(-y[1], 0.0))}

def create_controller(weights="policy.npz", **_):
    return PolicyController(weights)
```

`obs.lead_object()` and `obs.nearest_object()` are deliberately crude helpers —
a controller that wants better in-path reasoning should write it, since that
reasoning is part of what is being compared.

The built-in `linear_policy` controller is exactly this shape, with the same
seven features, and `LinearPolicyController.from_npz` loads weights from a file.
It is a working starting point for imitation or reinforcement learning; the
platform's run log (state, observation summary, command per tick) is the dataset.

Two practical notes:

* Set every seed you use from `context.seed`. The platform is otherwise
  bit-reproducible, and a controller that is not makes comparison meaningless.
* Compute time is measured and scored (`real_time_factor`). A policy that needs
  400 ms per tick is a legitimate result, and it will show in the score rather
  than being hidden by a wall-clock cut-off.

---

## 7. How the run is scored / 채점 방식

The platform records raw metrics and the user maps them to a score. Every
threshold and weight below is editable **while the platform runs**, because
mission time, safety and energy are quantities of different dimensions and there
is no single correct exchange rate between them.

임무 시간, 안전, 에너지는 서로 다른 차원의 값이므로 정답인 환산 비율은
존재하지 않습니다. 따라서 모든 임계값과 가중치는 **실행 중에** 조정 가능합니다.

Each metric is mapped linearly onto `[0, 100]`:

```
score = clip((bad - value) / (bad - good), 0, 1) * 100
```

`good` and `bad` may be in either order, so a metric where more is better
(clearance) and one where less is better (time) use the same formula. Infinity is
direction-aware: `min_clearance = inf` (nothing was near) scores 100, while
`time_to_goal = inf` (never finished) scores 0.

| Category | Metrics | Default category weight |
|---|---|---|
| **mission** 임무 | `time_to_goal`, `progress_ratio`, `mean_speed` | 1.0 |
| **safety** 안전 | `min_clearance`, `min_ttc`, `time_below_ttc`, `max_friction_usage`, `corridor_exit_time`, `red_light_violations` | 3.0 |
| **energy** 에너지 | `steering_effort`, `accel_effort`, `tractive_energy` | 1.0 |
| **comfort** 승차감 | `max_lat_accel`, `jerk_rms`, `cross_track_rms` | 0.7 |
| **compute** 연산 | `real_time_factor` | 0.3 |

The total is the weighted mean of the category scores, each of which is the
weighted mean of its metrics' scores.

A collision is handled separately by `collision_policy`: `"zero"` (default) makes
the total score 0 — a crash is not something a good time compensates for — or
`"penalty"` subtracts a fixed number of points, which is more useful when
ranking algorithms that all crash somewhere and you want to see which crashed
least.

---

## 8. Checklist before you submit a controller

- [ ] `control(obs)` returns in every branch — including when `obs.objects` is
      empty and when `obs.signal` is `None`.
- [ ] All per-run state is cleared in `reset(context)`.
- [ ] `steer` is normalized by `obs.vehicle.delta_max`, not by a hard-coded angle.
- [ ] Throttle and brake are never both positive unless you mean the brake.
- [ ] No `NaN` can escape: guard every division by speed, every `arctan2` of a
      zero vector, every gap that can be zero.
- [ ] Randomness is seeded from `context.seed`.
- [ ] The file runs on its own (`python my_controller.py` does nothing harmful) —
      the loader executes it at import.

---

## 9. Reference

* Machine-readable contract: `avsim/platform/controller_api.py`
* Worked templates: `examples/controllers/`
* Built-in controllers to read: `avsim/platform/controllers.py`
  (`mpc`, `pure_pursuit`, `linear_policy`)
* Scoring: `avsim/platform/scoring.py`
* Design decisions behind the models: `docs/design.md`
