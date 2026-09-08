"""The test platform: grid world, traffic, the plug-in contract, scoring, sessions."""

import json
import math

import numpy as np
import pytest

from avsim.core.geometry import polygon_distance, rect_corners

from avsim.platform.controller_api import (
    CONTRACT_VERSION,
    ControlCommand,
    Controller,
    ControllerLoadError,
    check_command,
    load_controller,
)
from avsim.platform.parameters import (
    ROAD_SURFACES,
    build_vehicle_params,
    clamp,
    default_values,
    derived_summary,
    merge,
    specs_json,
)
from avsim.platform.presets import PRESETS
from avsim.platform.scoring import DEFAULT_METRICS, MetricSpec, ScoreConfig, compare, score_run
from avsim.platform.session import RunConfig, RunSession
from avsim.world.grid import GridLayout, grid_network, node_id
from avsim.world.grid_traffic import ForcedVehicle, GridTrafficSource, TrafficConfig


# --- the grid road network ------------------------------------------------------

def test_grid_layout_is_a_well_of_roads():
    """The 우물 정자 layout: a 3x3 grid of signalized crossroads."""
    layout = grid_network(rows=3, cols=3, spacing=150.0)
    assert len(layout.centres) == 9
    assert layout.node(1, 1) == node_id(1, 1)

    # Nodes sit on a regular lattice, so the centre is the origin.
    centre = layout.centres[layout.node(1, 1)]
    assert centre == pytest.approx((0.0, 0.0), abs=1e-9)
    east = layout.centres[layout.node(2, 1)]
    assert east[0] - centre[0] == pytest.approx(150.0)

    # Every node carries both signal groups, and they are never green together.
    for node in layout.centres:
        ns, ew = f"{node}:NS", f"{node}:EW"
        assert ns in layout.signals.groups and ew in layout.signals.groups
        for t in np.linspace(0.0, 90.0, 400):
            states = {layout.signals.state(ns, t).value, layout.signals.state(ew, t).value}
            assert states != {"green"}


def test_route_through_the_grid_is_continuous_and_signalized():
    layout = grid_network()
    lanes = layout.route_lanes("n0_1", "E", ("straight", "left", "straight"), lane=0)
    path = layout.route_path(lanes)          # raises if the pieces do not connect

    # The route ends heading north after a single left turn.
    x, y, theta, _ = path.frames(np.array([path.length - 1.0]))
    assert math.degrees(float(theta[0])) == pytest.approx(90.0, abs=1e-6)

    # Every signalized stop line on the route is identified, in order.
    signals = layout.route_signals(lanes)
    assert [s for s, _, _ in signals] == sorted(s for s, _, _ in signals)
    assert all(group in layout.signals.groups for _, group, _ in signals)


def test_turn_lanes_are_respected():
    """With two lanes a right turn is only legal from the outside lane."""
    layout = grid_network(n_lanes=2)
    assert "right" in layout.allowed_manoeuvres(1)
    assert "right" not in layout.allowed_manoeuvres(0)
    assert "left" in layout.allowed_manoeuvres(0)
    with pytest.raises(ValueError):
        layout.route_lanes("n0_1", "E", ("right",), lane=0)


# --- traffic --------------------------------------------------------------------

def test_random_traffic_drives_without_colliding():
    """Other vehicles are not autonomous, but they do not drive into each other."""
    layout = grid_network()
    source = GridTrafficSource(layout, TrafficConfig(n_vehicles=14), seed=5)

    min_gap, stopped_seen, moving_seen = math.inf, False, False
    t = 0.0
    while t < 90.0:
        t += 0.1
        actors = source.step(t, 0.1)
        boxes = [rect_corners(a.x, a.y, a.psi, a.length, a.width) for a in actors]
        for i, a in enumerate(actors):
            moving_seen |= a.v > 3.0
            stopped_seen |= a.v < 0.05
            for j in range(i + 1, len(actors)):
                # Box-to-box, not centre-to-centre: two cars in adjacent lanes
                # are 3.5 m apart and perfectly fine.
                min_gap = min(min_gap, polygon_distance(boxes[i], boxes[j]))
    assert min_gap > 0.0, "traffic vehicles overlapped"
    assert moving_seen and stopped_seen, "traffic should both drive and wait"


def test_forced_vehicles_start_where_the_preset_puts_them():
    layout = grid_network()
    forced = ForcedVehicle(node="n1_1", direction="W", plan=("straight",), s0=30.0, v=8.0)
    source = GridTrafficSource(layout, TrafficConfig(n_vehicles=0), seed=1, forced=[forced])
    actors = source.step(0.1, 0.1)
    assert len(actors) == 1
    assert actors[0].v == pytest.approx(8.0, abs=1.5)


# --- the plug-in contract ---------------------------------------------------------

def test_command_normalization_matches_the_documented_scaling():
    veh = RunSession(RunConfig(preset="free_drive", duration=1.0)).vehicle_info()

    a, delta = ControlCommand(steer=0.5, throttle=1.0).to_physical(veh)
    assert delta == pytest.approx(0.5 * veh.delta_max)
    assert a == pytest.approx(veh.a_max)

    # Brake wins over throttle -- documented, not incidental.
    a, _ = ControlCommand(throttle=1.0, brake=0.5).to_physical(veh)
    assert a == pytest.approx(0.5 * veh.a_min)

    # from_physical is the exact inverse inside the limits.
    cmd = ControlCommand.from_physical(-2.0, -0.1, veh)
    a, delta = cmd.to_physical(veh)
    assert (a, delta) == pytest.approx((-2.0, -0.1), abs=1e-9)


def test_check_command_accepts_beginners_and_rejects_nan():
    assert check_command((0.2, 0.4)).throttle == pytest.approx(0.4)
    assert check_command((0.2, -0.4)).brake == pytest.approx(0.4)
    assert check_command({"steer": 3.0}).steer == pytest.approx(1.0)   # clipped
    with pytest.raises(ValueError):
        check_command({"steer": float("nan")})
    with pytest.raises(TypeError):
        check_command("hard left")


def test_plugin_loading_finds_a_controller_four_ways(tmp_path):
    def write(body):
        path = tmp_path / f"plug{abs(hash(body)) % 10**7}.py"
        path.write_text(body, encoding="utf-8")
        return path

    factory = write(
        "class C:\n"
        "    name = 'factory'\n"
        "    def control(self, obs):\n        return (0.0, 0.0)\n"
        "def create_controller(**kw):\n    return C()\n"
    )
    assert load_controller(factory).name == "factory"

    named = write("class Controller:\n    name='named'\n"
                  "    def control(self, obs):\n        return (0.0, 0.0)\n")
    assert load_controller(named).name == "named"

    instance = write("class C:\n    name='object'\n"
                     "    def control(self, obs):\n        return (0.0, 0.0)\n"
                     "controller = C()\n")
    assert load_controller(instance).name == "object"

    nothing = write("x = 1\n")
    with pytest.raises(ControllerLoadError):
        load_controller(nothing)

    broken = write("raise RuntimeError('boom')\n")
    with pytest.raises(ControllerLoadError):
        load_controller(broken)


def test_shipped_templates_load():
    for name in ("template_minimal", "template_stanley", "template_ml"):
        controller = load_controller(f"examples/controllers/{name}.py")
        assert callable(controller.control)


# --- parameters --------------------------------------------------------------------

def test_parameter_defaults_reproduce_the_reference_vehicle():
    d = derived_summary(build_vehicle_params(default_values()))
    assert d["wheelbase"] == pytest.approx(2.70, abs=1e-9)
    assert d["understeer_gradient"] == pytest.approx(3.75e-3, rel=1e-3)
    assert d["characteristic_speed"] == pytest.approx(26.8, abs=0.1)
    assert d["balance"] == "understeer"


def test_parameters_clamp_and_merge():
    values = clamp(merge({"mu": 99.0, "m": -5.0}))
    spec = {s["key"]: s for s in specs_json()}
    assert values["mu"] <= spec["mu"]["max"]
    assert values["m"] >= spec["m"]["min"]
    # An unknown key is not silently accepted into the vehicle build.
    assert "nonsense" not in clamp(merge({"nonsense": 1.0}))


def test_low_friction_lowers_the_lateral_limit():
    dry = derived_summary(build_vehicle_params(merge({"mu": 0.9})))
    ice = derived_summary(build_vehicle_params(merge({"mu": 0.15})))
    assert ice["max_lateral_accel"] < 0.2 * dry["max_lateral_accel"]
    assert dict(ROAD_SURFACES)["Ice (빙판)"] == pytest.approx(0.15)


# --- scoring -------------------------------------------------------------------------

def test_metric_scoring_is_direction_aware_including_infinity():
    slower_is_better = MetricSpec("t", "t", "mission", "s", good=20.0, bad=90.0)
    assert slower_is_better.score(20.0) == pytest.approx(100.0)
    assert slower_is_better.score(90.0) == pytest.approx(0.0)
    assert slower_is_better.score(55.0) == pytest.approx(50.0)
    assert slower_is_better.score(float("inf")) == 0.0     # never finished

    more_is_better = MetricSpec("c", "c", "safety", "m", good=3.0, bad=0.2)
    assert more_is_better.score(3.0) == pytest.approx(100.0)
    assert more_is_better.score(0.2) == pytest.approx(0.0)
    assert more_is_better.score(float("inf")) == 100.0     # nothing was near


def test_weights_change_the_total_without_rerunning():
    raw = {
        "collisions": 0, "time_to_goal": 40.0, "progress_ratio": 1.0, "mean_speed": 9.0,
        "min_clearance": 1.0, "min_ttc": 2.0, "time_below_ttc": 1.0,
        "max_friction_usage": 0.8, "corridor_exit_time": 0.0, "red_light_violations": 0.0,
        "steering_effort": 6.0, "accel_effort": 30.0, "tractive_energy": 700.0,
        "max_lat_accel": 3.0, "jerk_rms": 2.0, "cross_track_rms": 0.3,
        "real_time_factor": 0.4,
    }
    base = ScoreConfig()
    safety_first = ScoreConfig()
    safety_first.category_weights["safety"] = 20.0

    a, b = score_run(raw, base), score_run(raw, safety_first)
    assert a.total != pytest.approx(b.total)
    assert b.total == pytest.approx(
        b.categories["safety"], abs=abs(b.total - b.categories["safety"])
    )
    # The category scores themselves do not depend on the category weights.
    assert a.categories["safety"] == pytest.approx(b.categories["safety"])


def test_collision_policy():
    raw = {"collisions": 1, "time_to_goal": 30.0, "min_clearance": 0.0, "progress_ratio": 1.0}
    zeroed = score_run(raw, ScoreConfig(collision_policy="zero"))
    assert zeroed.total == 0.0 and zeroed.collided

    penalised = score_run(raw, ScoreConfig(collision_policy="penalty", collision_penalty=30.0))
    clean = score_run({**raw, "collisions": 0}, ScoreConfig(collision_policy="penalty"))
    assert penalised.total == pytest.approx(max(clean.total - 30.0, 0.0), abs=1e-9)


def test_score_config_round_trips_through_json():
    cfg = ScoreConfig()
    cfg.metrics[0].weight = 3.5
    cfg.category_weights["energy"] = 2.5
    cfg.ttc_threshold = 1.25
    restored = ScoreConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.metrics[0].weight == 3.5
    assert restored.category_weights["energy"] == 2.5
    assert restored.ttc_threshold == 1.25


def test_compare_ranks_best_first():
    raw = {"collisions": 0, "progress_ratio": 1.0, "time_to_goal": 30.0}
    good = score_run(raw, ScoreConfig())
    bad = score_run({**raw, "time_to_goal": 85.0}, ScoreConfig())
    ranking = compare([("slow", bad), ("quick", good)])
    assert [r["name"] for r in ranking] == ["quick", "slow"]
    assert ranking[0]["rank"] == 1


# --- presets and sessions -------------------------------------------------------------

def test_every_preset_builds_a_connected_route():
    for key, preset in PRESETS.items():
        layout = preset.build_layout()
        lanes = layout.route_lanes(preset.ego.node, preset.ego.direction,
                                   preset.ego.plan, preset.ego.lane)
        path = layout.route_path(lanes)      # raises on a discontinuity
        assert path.length > 20.0, key
        assert preset.duration > 0.0


def test_scenario_presets_cover_the_required_situations():
    assert {"unprotected_left", "right_turn", "overtake_straight"} <= set(PRESETS)
    assert PRESETS["unprotected_left"].forced, "an unprotected left needs oncoming traffic"
    assert PRESETS["overtake_straight"].grid.n_lanes >= 2, "overtaking needs a second lane"


def test_a_short_session_runs_scores_and_is_reproducible():
    def run():
        config = RunConfig(preset="free_drive", controller="pure_pursuit",
                           seed=2, duration=12.0, n_vehicles=0)
        return RunSession(config).run()

    first, second = run(), run()
    assert first.metrics["distance"] > 20.0
    assert first.metrics["collisions"] == 0
    assert 0.0 <= first.score.total <= 100.0
    assert set(first.score.categories) >= {"mission", "safety", "energy"}
    # Same configuration, same answer -- the property every comparison rests on.
    assert first.metrics["distance"] == pytest.approx(second.metrics["distance"], abs=1e-12)
    assert first.score.total == pytest.approx(second.score.total, abs=1e-12)


def test_a_controller_that_raises_is_a_result_not_a_crash():
    class Broken(Controller):
        name = "broken"

        def control(self, obs):
            raise ZeroDivisionError("the author's bug, not the platform's")

    session = RunSession(RunConfig(preset="free_drive", duration=2.0, n_vehicles=0))
    session.controller = Broken()
    result = session.run()
    assert result.metrics["control_ticks"] > 5
    assert any("controller_error" in row for row in session.log)


def test_the_observation_carries_the_documented_contract():
    session = RunSession(RunConfig(preset="unprotected_left", duration=3.0, seed=1))
    session.step()
    frame = session.frames[-1]
    assert set(frame.readout) >= {"speed", "progress", "e_y", "min_clearance", "compute_ms"}
    assert set(frame.command) == {"steer", "throttle", "brake"}

    scene = session.static_scene()
    assert scene["lanes"] and scene["nodes"] and scene["route"]
    assert CONTRACT_VERSION == "1.0"
