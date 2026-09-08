"""The adjustable parameter set, described once so the UI can build itself.

Every knob the platform exposes is a :class:`ParamSpec`: a key, a range, a
default, a unit and the group it belongs to.  The web UI reads this list and
renders a slider plus a number box for each; nothing about the UI knows what a
cornering stiffness is.

Adding a parameter is therefore a one-line change here, and
:func:`build_vehicle_params` is the single place where a dictionary of values
becomes a :class:`~avsim.models.params.VehicleParams`.  Keeping that conversion
in one function is what stops the UI and the simulator from disagreeing about
what "mu" means.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from ..models.params import (
    ActuatorParams,
    SuspensionParams,
    TireParams,
    VehicleParams,
)


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    group: str
    default: Any
    kind: str = "float"           #: "float" | "int" | "choice" | "bool"
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    choices: tuple[tuple[str, Any], ...] = ()
    help: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["choices"] = [{"label": a, "value": b} for a, b in self.choices]
        return d


#: Road surfaces, as a friction coefficient.  Offered as a list because "wet
#: asphalt" is the question a user actually has, and 0.6 is the answer.
ROAD_SURFACES = (
    ("Dry asphalt (마른 아스팔트)", 0.90),
    ("Wet asphalt (젖은 노면)", 0.60),
    ("Packed snow (다져진 눈)", 0.35),
    ("Ice (빙판)", 0.15),
    ("Race compound (레이싱)", 1.20),
)

PARAMETER_SPECS: tuple[ParamSpec, ...] = (
    # --- vehicle body ---------------------------------------------------------
    ParamSpec("m", "Mass 질량", "vehicle", 1500.0, min=800.0, max=3000.0, step=10.0, unit="kg"),
    ParamSpec("I_z", "Yaw inertia 요 관성", "vehicle", 2400.0, min=800.0, max=6000.0, step=50.0,
              unit="kg·m²"),
    ParamSpec("l_f", "CG to front axle 전축거리", "vehicle", 1.2, min=0.8, max=2.0, step=0.01,
              unit="m", help="l_f + l_r is the wheelbase L."),
    ParamSpec("l_r", "CG to rear axle 후축거리", "vehicle", 1.5, min=0.8, max=2.2, step=0.01,
              unit="m"),
    ParamSpec("length", "Body length 전장", "vehicle", 4.6, min=3.0, max=6.0, step=0.05, unit="m"),
    ParamSpec("width", "Body width 전폭", "vehicle", 1.85, min=1.4, max=2.4, step=0.01, unit="m"),

    # --- tires ------------------------------------------------------------------
    ParamSpec("surface", "Road surface 노면", "tire", 0.90, kind="choice", choices=ROAD_SURFACES,
              unit="mu", help="Sets the tire-road friction coefficient."),
    ParamSpec("mu", "Friction coefficient 마찰계수", "tire", 0.90, min=0.10, max=1.30, step=0.01,
              help="Overrides the surface list when moved."),
    ParamSpec("C_f", "Front cornering stiffness 전륜 코너링강성", "tire", 80000.0,
              min=20000.0, max=200000.0, step=1000.0, unit="N/rad"),
    ParamSpec("C_r", "Rear cornering stiffness 후륜 코너링강성", "tire", 100000.0,
              min=20000.0, max=200000.0, step=1000.0, unit="N/rad"),
    ParamSpec("alpha_peak_deg", "Peak slip angle 최대 슬립각", "tire", 8.0, min=3.0, max=16.0,
              step=0.1, unit="deg", help="Where the Magic Formula peaks."),
    ParamSpec("tire_model", "Tire model 타이어 모델", "tire", "pacejka", kind="choice",
              choices=(("Magic Formula (포화 포함)", "pacejka"), ("Linear (선형)", "linear")),
              help="Linear tires reproduce the understeer gradient exactly; "
                   "Pacejka adds saturation."),

    # --- steering and actuators ----------------------------------------------------
    ParamSpec("delta_max_deg", "Max steer angle 최대 조향각", "steering", 35.0, min=15.0, max=50.0,
              step=0.5, unit="deg", help="Road-wheel angle, not steering-wheel."),
    ParamSpec("delta_rate_max_deg", "Max steer rate 최대 조향속도", "steering", 40.0,
              min=10.0, max=120.0, step=1.0, unit="deg/s"),
    ParamSpec("tau_steer", "Steering lag 조향 지연", "steering", 0.10, min=0.02, max=0.40,
              step=0.01, unit="s"),
    ParamSpec("a_max", "Max acceleration 최대 가속", "steering", 3.0, min=1.0, max=6.0, step=0.1,
              unit="m/s²"),
    ParamSpec("a_min", "Max deceleration 최대 감속", "steering", -8.0, min=-12.0, max=-2.0,
              step=0.1, unit="m/s²"),
    ParamSpec("tau_drive", "Powertrain lag 구동 지연", "steering", 0.20, min=0.02, max=0.60,
              step=0.01, unit="s"),
    ParamSpec("tau_brake", "Brake lag 제동 지연", "steering", 0.08, min=0.02, max=0.40,
              step=0.01, unit="s"),

    # --- powertrain -------------------------------------------------------------------
    ParamSpec("drive_layout", "Drive layout 구동방식", "powertrain", "fwd", kind="choice",
              choices=(("Front-wheel drive (전륜)", "fwd"), ("Rear-wheel drive (후륜)", "rwd"),
                       ("All-wheel drive (사륜)", "awd"))),
    ParamSpec("max_drive_force", "Peak drive force 최대 구동력", "powertrain", 6000.0,
              min=2000.0, max=20000.0, step=100.0, unit="N"),
    ParamSpec("max_brake_force", "Peak brake force 최대 제동력", "powertrain", 14000.0,
              min=5000.0, max=40000.0, step=100.0, unit="N"),
    ParamSpec("brake_bias_front", "Brake bias (front) 전륜 제동배분", "powertrain", 0.65,
              min=0.30, max=0.90, step=0.01),
    ParamSpec("wheel_radius", "Wheel radius 휠 반경", "powertrain", 0.32, min=0.25, max=0.45,
              step=0.005, unit="m"),
    ParamSpec("c_drag", "Aero drag 공기저항", "powertrain", 0.40, min=0.0, max=1.5, step=0.01,
              unit="N·s²/m²"),
    ParamSpec("c_rr", "Rolling resistance 구름저항", "powertrain", 0.013, min=0.0, max=0.05,
              step=0.001),

    # --- suspension ---------------------------------------------------------------------
    ParamSpec("h_cg", "CG height 무게중심 높이", "suspension", 0.55, min=0.30, max=0.90, step=0.01,
              unit="m", help="Drives longitudinal and lateral load transfer."),
    ParamSpec("track", "Track width 윤거", "suspension", 1.60, min=1.20, max=2.10, step=0.01,
              unit="m"),
    ParamSpec("K_roll_f", "Front roll stiffness 전륜 롤강성", "suspension", 60000.0,
              min=10000.0, max=200000.0, step=1000.0, unit="N·m/rad"),
    ParamSpec("K_roll_r", "Rear roll stiffness 후륜 롤강성", "suspension", 40000.0,
              min=10000.0, max=200000.0, step=1000.0, unit="N·m/rad"),
    ParamSpec("C_roll", "Roll damping 롤 감쇠", "suspension", 6000.0, min=500.0, max=30000.0,
              step=100.0, unit="N·m·s/rad"),
    ParamSpec("K_pitch", "Pitch stiffness 피치강성", "suspension", 130000.0, min=20000.0,
              max=400000.0, step=1000.0, unit="N·m/rad"),
    ParamSpec("C_pitch", "Pitch damping 피치 감쇠", "suspension", 14000.0, min=1000.0, max=60000.0,
              step=500.0, unit="N·m·s/rad"),
    ParamSpec("suspension_dynamic", "Dynamic suspension 동적 현가", "suspension", True, kind="bool",
              help="Off = quasi-static load transfer with no lag."),

    # --- simulation ------------------------------------------------------------------------
    ParamSpec("control_dt", "Control period 제어 주기", "simulation", 0.10, min=0.02, max=0.30,
              step=0.01, unit="s"),
    ParamSpec("plant_dt", "Plant step 플랜트 적분 스텝", "simulation", 0.02, min=0.005, max=0.05,
              step=0.005, unit="s"),
    ParamSpec("speed_limit", "Speed limit 제한속도", "simulation", 13.9, min=5.0, max=30.0,
              step=0.1, unit="m/s"),
    ParamSpec("mpc_horizon", "MPC horizon MPC 지평", "simulation", 25, kind="int", min=10, max=45,
              step=1, unit="steps", help="Built-in MPC only."),
    ParamSpec("mpc_dt", "MPC step MPC 스텝", "simulation", 0.12, min=0.05, max=0.30, step=0.01,
              unit="s", help="Built-in MPC only."),
)

PARAMETER_GROUPS = (
    ("vehicle", "Vehicle body 차체"),
    ("tire", "Tires & road 타이어·노면"),
    ("steering", "Steering & actuators 조향·액추에이터"),
    ("powertrain", "Powertrain 구동계"),
    ("suspension", "Suspension 현가장치"),
    ("simulation", "Simulation 시뮬레이션"),
)

_BY_KEY = {p.key: p for p in PARAMETER_SPECS}


def default_values() -> dict:
    """Every parameter at its default."""
    return {p.key: p.default for p in PARAMETER_SPECS}


def merge(values: dict | None) -> dict:
    """Defaults overlaid with ``values``, ignoring keys that are not parameters."""
    out = default_values()
    for k, v in (values or {}).items():
        if k in _BY_KEY:
            out[k] = v
    return out


def clamp(values: dict) -> dict:
    """Clip numeric parameters into their declared range.

    The UI already limits its sliders; this exists for values arriving over the
    API, where a typo in a script would otherwise reach the plant as a mass of
    ``-1`` and fail somewhere far away with a message about a Cholesky
    factorization.
    """
    out = dict(values)
    for key, spec in _BY_KEY.items():
        if key not in out or spec.kind in ("choice", "bool"):
            continue
        v = float(out[key])
        if spec.min is not None:
            v = max(v, spec.min)
        if spec.max is not None:
            v = min(v, spec.max)
        out[key] = int(round(v)) if spec.kind == "int" else v
    return out


def build_vehicle_params(values: dict | None = None) -> VehicleParams:
    """Turn a parameter dictionary into a :class:`VehicleParams`.

    The one place the UI's names become the simulator's. Angles arrive in
    degrees because that is how they are read on a slider, and are converted
    here -- the rest of the package is radians throughout.
    """
    v = clamp(merge(values))

    tire = TireParams(mu=float(v["mu"]))
    actuator = ActuatorParams(
        delta_max=float(np.deg2rad(v["delta_max_deg"])),
        delta_rate_max=float(np.deg2rad(v["delta_rate_max_deg"])),
        tau_steer=float(v["tau_steer"]),
        a_max=float(v["a_max"]),
        a_min=float(v["a_min"]),
        tau_drive=float(v["tau_drive"]),
        tau_brake=float(v["tau_brake"]),
    )
    suspension = SuspensionParams(
        h_cg=float(v["h_cg"]),
        h_roll=float(v["h_cg"]) * 0.82,
        h_pitch=float(v["h_cg"]) * 0.91,
        track=float(v["track"]),
        K_roll_f=float(v["K_roll_f"]),
        K_roll_r=float(v["K_roll_r"]),
        C_roll=float(v["C_roll"]),
        K_pitch=float(v["K_pitch"]),
        C_pitch=float(v["C_pitch"]),
        dynamic=bool(v["suspension_dynamic"]),
    )
    return VehicleParams(
        m=float(v["m"]),
        I_z=float(v["I_z"]),
        l_f=float(v["l_f"]),
        l_r=float(v["l_r"]),
        C_f=float(v["C_f"]),
        C_r=float(v["C_r"]),
        length=float(v["length"]),
        width=float(v["width"]),
        wheel_radius=float(v["wheel_radius"]),
        drive_layout=str(v["drive_layout"]),
        brake_bias_front=float(v["brake_bias_front"]),
        max_drive_force=float(v["max_drive_force"]),
        max_brake_force=float(v["max_brake_force"]),
        c_drag=float(v["c_drag"]),
        c_rr=float(v["c_rr"]),
        tire=tire,
        actuator=actuator,
        suspension=suspension,
    )


def derived_summary(params: VehicleParams) -> dict:
    """Quantities worth showing next to the sliders because they move with them.

    A user changing ``C_f`` wants to see the understeer gradient change; a user
    lowering ``mu`` wants to see the lateral limit fall.  Showing the derived
    numbers is how a parameter panel teaches rather than merely accepting input.
    """
    return {
        "wheelbase": params.L,
        "understeer_gradient": params.understeer_gradient,
        "characteristic_speed": params.characteristic_speed,
        "critical_speed": params.critical_speed,
        "max_lateral_accel": params.max_lateral_accel(1.0),
        "planning_lateral_accel": params.max_lateral_accel(0.5),
        "min_turn_radius": params.L / float(np.tan(params.actuator.delta_max)),
        "static_load_front": params.F_z_front_static,
        "static_load_rear": params.F_z_rear_static,
        "balance": (
            "understeer" if params.understeer_gradient > 0 else "oversteer"
        ),
    }


def specs_json() -> list[dict]:
    return [p.to_json() for p in PARAMETER_SPECS]
