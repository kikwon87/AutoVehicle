"""Rendering: a top-down view of the world and the time series behind a KPI.

Two entry points:

:func:`plot_run`
    A static multi-panel figure -- the trajectory over the road, plus the
    signals a KPI is computed from.  Reading the number and the trace together
    is the only way to tell a 0.4 m RMS made of one excursion from a 0.4 m RMS
    made of a permanent bias, and they call for different fixes.

:func:`animate_run`
    A GIF of the same scene with the planned trajectory, the MPC prediction and
    the perceived tracks drawn each frame.  Slower, and worth it when a KPI
    fails for a reason no scalar explains.

Matplotlib is an optional dependency; importing this module without it raises
with an instruction rather than failing deep inside a draw call.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:  # pragma: no cover - import guard
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Polygon as MplPolygon
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "avsim.viz needs matplotlib; install it with `pip install 'avsim[viz]'`"
    ) from exc

from ..eval.runner import RunResult
from ..world.network import RoadNetwork

EGO_COLOR = "#1f6feb"
ACTOR_COLOR = "#d1242f"
PLAN_COLOR = "#2da44e"
PRED_COLOR = "#bf8700"


def _draw_network(ax, net: RoadNetwork) -> None:
    for lane in net.lanes.values():
        left, right = lane.edges(ds=2.0)
        style = "--" if lane.kind.startswith("connector") else "-"
        ax.plot(left[:, 0], left[:, 1], style, color="0.75", lw=0.8, zorder=0)
        ax.plot(right[:, 0], right[:, 1], style, color="0.75", lw=0.8, zorder=0)
        c = lane.sample(ds=2.0)
        ax.plot(c[:, 0], c[:, 1], ":", color="0.88", lw=0.7, zorder=0)
    if net.box_half is not None:
        b = net.box_half
        ax.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], color="0.6", lw=0.9, zorder=0)


def _vehicle_patch(state, color, alpha=0.85):
    return MplPolygon(state.corners(), closed=True, facecolor=color,
                      edgecolor="black", lw=0.6, alpha=alpha, zorder=3)


def plot_run(result: RunResult, path: str | None = None, stride: int = 25, figsize=(15, 10)):
    """Static summary figure for one run."""
    hist = result.history
    tel = result.telemetry
    setup = result.setup
    t = np.array([h.t for h in hist])

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(4, 3)

    # --- top-down ------------------------------------------------------------
    ax = fig.add_subplot(gs[0:2, :])
    _draw_network(ax, setup.world.network)
    route = setup.route.sample(ds=1.0)
    ax.plot(route[:, 0], route[:, 1], color=PLAN_COLOR, lw=1.4, alpha=0.6, label="route")
    ego_xy = np.array([[h.ego[0], h.ego[1]] for h in hist])
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], color=EGO_COLOR, lw=1.6, label="ego path")
    for h in hist[::stride]:
        ax.add_patch(_vehicle_patch(h.ego_state, EGO_COLOR, 0.35))
        for a in h.actors:
            ax.add_patch(_vehicle_patch(a, ACTOR_COLOR, 0.25))
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right", fontsize=8)
    verdict = "PASS" if result.passed else "FAIL"
    ax.set_title(f"{setup.name} -- {verdict}\n{setup.description}", fontsize=10)
    pad = 12.0
    ax.set_xlim(ego_xy[:, 0].min() - pad, ego_xy[:, 0].max() + pad)
    ax.set_ylim(ego_xy[:, 1].min() - pad, ego_xy[:, 1].max() + pad)

    # --- time series ---------------------------------------------------------
    e_y = np.array(result.report.extra["e_y"])
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, e_y, color=EGO_COLOR)
    hw = setup.stack.cfg.corridor_half_width
    ax.axhline(hw, color="0.6", ls="--", lw=0.8)
    ax.axhline(-hw, color="0.6", ls="--", lw=0.8)
    ax.set_ylabel("cross-track [m]")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, [h.ego_state.v for h in hist], color=EGO_COLOR, label="speed")
    if tel:
        ax.plot([r.t for r in tel], [r.target_speed for r in tel], color=PLAN_COLOR,
                ls="--", lw=1.0, label="target")
    ax.set_ylabel("speed [m/s]")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t, [h.diagnostics["a_x"] for h in hist], label="$a_x$")
    ax.plot(t, [h.diagnostics["a_y"] for h in hist], label="$a_y$")
    ax.set_ylabel("accel [m/s$^2$]")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, np.rad2deg([h.diagnostics["delta_actual"] for h in hist]), label=r"$\delta$")
    if tel:
        ax.plot([r.t for r in tel], np.rad2deg([r.delta_cmd for r in tel]), ls="--", lw=0.9,
                label=r"$\delta_{cmd}$")
    ax.set_ylabel("steer [deg]")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, [max(h.diagnostics["usage_front"], h.diagnostics["usage_rear"]) for h in hist],
            color="#8250df")
    ax.axhline(1.0, color=ACTOR_COLOR, ls="--", lw=0.8)
    ax.set_ylabel("friction usage [-]")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[3, 2])
    if tel:
        ax.plot([r.t for r in tel], [r.mpc_time * 1e3 for r in tel], color="#953800", lw=0.9)
        ax.axhline(setup.stack.cfg.control_dt * 1e3, color="0.5", ls="--", lw=0.8,
                   label="control period")
        ax.legend(fontsize=7)
    ax.set_ylabel("MPC solve [ms]")
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)

    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    return fig


def plot_behavior_timeline(result: RunResult, path: str | None = None, figsize=(12, 2.4)):
    """A colour bar of the behaviour state over time, with the reasons annotated."""
    tel = result.telemetry
    if not tel:
        raise ValueError("no telemetry to plot")
    states = sorted({r.behavior for r in tel})
    cmap = plt.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(states)}

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    t = [r.t for r in tel]
    dt = (t[1] - t[0]) if len(t) > 1 else 0.1
    for r in tel:
        ax.barh(0, dt, left=r.t, height=0.6, color=colors[r.behavior], edgecolor="none")
    prev = None
    for r in tel:
        if r.behavior != prev:
            ax.text(r.t, 0.42, f"{r.behavior}\n{r.reason}", fontsize=6, rotation=0,
                    va="bottom", ha="left")
            prev = r.behavior
    ax.set_yticks([])
    ax.set_xlabel("t [s]")
    ax.set_title(f"{result.setup.name}: behaviour timeline", fontsize=10)
    ax.set_ylim(-0.5, 1.6)
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    return fig


def animate_run(
    result: RunResult,
    path: str,
    fps: int = 20,
    stride: int = 5,
    window: float = 55.0,
):
    """Write a GIF of the run, following the ego."""
    hist = result.history[::stride]
    setup = result.setup
    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    _draw_network(ax, setup.world.network)
    route = setup.route.sample(ds=1.0)
    ax.plot(route[:, 0], route[:, 1], color=PLAN_COLOR, lw=1.2, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    trail, = ax.plot([], [], color=EGO_COLOR, lw=1.3)
    patches: list = []
    title = ax.set_title("")

    def update(i):
        nonlocal patches
        for p in patches:
            p.remove()
        patches = []
        h = hist[i]
        patches.append(ax.add_patch(_vehicle_patch(h.ego_state, EGO_COLOR)))
        for a in h.actors:
            patches.append(ax.add_patch(_vehicle_patch(a, ACTOR_COLOR)))
        xy = np.array([[g.ego[0], g.ego[1]] for g in hist[: i + 1]])
        trail.set_data(xy[:, 0], xy[:, 1])
        ax.set_xlim(h.ego[0] - window / 2, h.ego[0] + window / 2)
        ax.set_ylim(h.ego[1] - window / 2, h.ego[1] + window / 2)
        sig = "  ".join(f"{k}:{v}" for k, v in h.signals.items())
        title.set_text(f"{setup.name}  t = {h.t:5.2f}s  v = {h.ego_state.v:5.2f} m/s   {sig}")
        return patches + [trail, title]

    anim = animation.FuncAnimation(fig, update, frames=len(hist), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_kpi_matrix(reports: Sequence, path: str | None = None, figsize=(12, 6)):
    """A pass/fail grid over scenarios and KPIs -- the suite at a glance."""
    names = sorted({k.name for r in reports for k in r.kpis})
    grid = np.full((len(reports), len(names)), np.nan)
    for i, r in enumerate(reports):
        for k in r.kpis:
            if k.threshold is not None:
                grid[i, names.index(k.name)] = 1.0 if k.passed else 0.0

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(reports)))
    ax.set_yticklabels([r.scenario for r in reports], fontsize=8)
    ax.set_title("KPI pass/fail matrix (grey = no threshold)", fontsize=10)
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    return fig
