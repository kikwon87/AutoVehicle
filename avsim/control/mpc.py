"""Nonlinear model predictive control for the vehicle.

The optimal-control problem, in the form of Lecture 2/3's *Recall: Where the
Model Enters*::

    min  sum_{k=0}^{N-1} q(x_k, u_k) + p(x_N)
    s.t. x_{k+1} = F_h(x_k, u_k)
         (x_k, u_k) in C

with the concrete choices:

**State** ``x = [X, Y, psi, v, delta]`` -- the rear-axle kinematic bicycle with
the **steering angle as a state**.

**Input** ``u = [a, delta_dot]`` -- acceleration and steering *rate*.

Making the rate the input is the whole trick.  The lecture's running example
lists ``|delta_k - delta_{k-1}| <= delta_dot_max h`` as a constraint to be
added; here it is not a constraint at all but a box on the input, which the
backward pass enforces exactly and for free.  The same move makes the steering
command continuous by construction, so the plant's rate-limited actuator is
never asked for a step it cannot follow.

**Discretization** RK4 on the continuous field, with Jacobians taken *through*
the RK4 step by :func:`avsim.models.linearization.rk4_jacobians`, so the
derivatives match the discrete constraint the solver evaluates.

**Constraints** steering limit, speed range, lateral-acceleration budget, the
lane corridor as two affine half-spaces per stage, and obstacle avoidance as
oriented-ellipse clearances between multi-circle body covers.  All of them are
handled by the augmented Lagrangian in :mod:`avsim.control.ilqr`.

**Model regime.**  The prediction model is kinematic, which the lecture endorses
for parking and urban planning *provided the lateral acceleration stays inside
the envelope* -- which is exactly why ``a_y`` is an explicit constraint here
rather than an assumption.  The plant it drives is the saturating dynamic
bicycle, so the mismatch is real and is measured by the KPI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.conventions import wrap_to_pi
from ..core.geometry import multi_circle_cover
from ..models.linearization import rk4_jacobians
from ..models.params import VehicleParams
from ..planning.prediction import Prediction
from .ilqr import ILQR, ILQROptions

NX, NU = 5, 2
IX, IY, IPSI, IV, IDELTA = 0, 1, 2, 3, 4
IA, IDDELTA = 0, 1


@dataclass
class CorridorStage:
    """Lane boundary at one stage, as two affine half-spaces.

    ``lo <= (p - p_ref) . n <= hi``.  This is the Frenet box constraint, frozen
    at the reference point for this stage -- the lecture's "constraints
    linearize too, and that is the point".  It is exact for a straight lane and
    a first-order approximation on a curved one, refreshed every MPC step.
    """

    p_ref: np.ndarray  #: (2,) point on the lane centerline
    normal: np.ndarray  #: (2,) unit left normal
    lo: float
    hi: float


@dataclass
class StopLineConstraint:
    """A half-space the vehicle's front bumper must not cross.

    ``(p_front - point) . tangent <= 0``

    A stop is not reliably produced by a speed reference alone: the reference
    says *how fast* to be at each time, and any tracking lag turns into
    overshoot past the line.  Stating the stop as a position constraint makes
    the optimizer responsible for it, and the augmented Lagrangian then trades
    comfort against it rather than ignoring it.
    """

    point: np.ndarray   #: (2,) world point on the stop line
    tangent: np.ndarray  #: (2,) unit vector along the direction of travel


@dataclass
class MPCConfig:
    horizon: int = 25
    dt: float = 0.12
    #: State weights, in the **path frame**: ``[e_lon, e_lat, e_psi, e_v, delta]``.
    #:
    #: Weighting ``X`` and ``Y`` directly would make the lateral gain depend on
    #: which way the road happens to point -- 6 on a road running north, 2 on
    #: one running east, and anything between on a curve. Rotating the position
    #: error into the reference tangent/normal makes "lateral" mean lateral
    #: everywhere, which is the difference between holding a curve and drifting
    #: 1.3 m wide through it.
    Q: tuple[float, ...] = (1.0, 12.0, 15.0, 2.0, 0.5)
    Qf: tuple[float, ...] = (2.0, 24.0, 30.0, 4.0, 0.5)
    #: input weights on ``[a, delta_dot]``
    R: tuple[float, ...] = (0.6, 20.0)
    v_max: float = 30.0
    v_min: float = 0.0
    a_y_max: float = 4.4
    #: fraction of the steering limit the optimizer may use
    delta_margin: float = 0.95
    #: Replace the kinematic yaw equation ``psi' = v tan(delta) / L`` with
    #: ``psi' = v tan(delta) / (L + K_us v^2)``.
    #:
    #: The two agree at low speed and diverge exactly as the lecture's
    #: steady-state relation ``delta_ss = (L + K_us V^2) kappa`` says they must.
    #: Without it the prediction model has no understeer at all, so on a curve
    #: the optimizer commands the geometric steering angle, the real vehicle
    #: runs wide, and the feedback is left to generate a steady-state demand the
    #: model already knows about -- which is precisely what the lecture warns
    #: against. Measured on a 300 m radius at 16 m/s, this is the difference
    #: between a steady 1.9 m drift and holding the lane.
    understeer_correction: bool = True
    #: how many sigma of prediction uncertainty the obstacle ellipse clears
    inflate_sigma: float = 1.0
    n_circles: int = 3
    max_obstacles: int = 4
    #: solver budget; the controller runs at ``1 / dt`` Hz, so this must be less
    time_budget: float = 0.030
    max_iter: int = 10
    max_al_iter: int = 4


@dataclass
class MPCResult:
    a_cmd: float
    delta_cmd: float
    X: np.ndarray
    U: np.ndarray
    cost: float
    violation: float
    status: str
    solve_time: float
    iterations: int
    feasible: bool
    predicted_a_y: np.ndarray = field(default_factory=lambda: np.zeros(0))


class VehicleMPC:
    """Receding-horizon nonlinear MPC on the rear-axle kinematic bicycle."""

    def __init__(self, params: VehicleParams, config: MPCConfig | None = None):
        self.p = params
        self.cfg = config or MPCConfig()
        self.cfg.a_y_max = min(self.cfg.a_y_max, params.max_lateral_accel(1.0))
        self._ego_off, self._ego_r = multi_circle_cover(
            params.length, params.width, self.cfg.n_circles
        )
        self._ego_body_offset = 0.5 * params.length - params.rear_overhang

        self._U_prev: np.ndarray | None = None
        self._lam_prev: np.ndarray | None = None
        self._penalty_prev: float | None = None

        # Per-solve context, refreshed by solve() before the solver runs.
        self._ref = np.zeros((self.cfg.horizon + 1, 4))
        self._ref_cos = np.ones(self.cfg.horizon + 1)
        self._ref_sin = np.zeros(self.cfg.horizon + 1)
        self._corridor: list[CorridorStage | None] = [None] * (self.cfg.horizon + 1)
        self._obs: list[list[tuple[np.ndarray, float, float, float]]] = [
            [] for _ in range(self.cfg.horizon + 1)
        ]
        self._nc = 0
        self._stop: StopLineConstraint | None = None
        #: rear axle to front bumper [m]
        self._front_offset = params.length - params.rear_overhang

    # --- model ---------------------------------------------------------------

    def _effective_wheelbase(self, v: float) -> float:
        """``L`` or ``L + K_us v^2``, depending on ``understeer_correction``."""
        if not self.cfg.understeer_correction:
            return self.p.L
        return self.p.L + self.p.understeer_gradient * v * v

    def field(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        psi, v, delta = x[IPSI], x[IV], x[IDELTA]
        Le = self._effective_wheelbase(v)
        return np.array(
            [v * np.cos(psi), v * np.sin(psi), v * np.tan(delta) / Le, u[IA], u[IDDELTA]]
        )

    def _A_c(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        psi, v, delta = x[IPSI], x[IV], x[IDELTA]
        A = np.zeros((NX, NX))
        A[IX, IPSI] = -v * np.sin(psi)
        A[IX, IV] = np.cos(psi)
        A[IY, IPSI] = v * np.cos(psi)
        A[IY, IV] = np.sin(psi)
        Le = self._effective_wheelbase(v)
        if self.cfg.understeer_correction:
            # d/dv [ v tan d / (L + K v^2) ] = tan d (L - K v^2) / (L + K v^2)^2
            K = self.p.understeer_gradient
            A[IPSI, IV] = np.tan(delta) * (self.p.L - K * v * v) / Le**2
        else:
            A[IPSI, IV] = np.tan(delta) / Le
        A[IPSI, IDELTA] = v / (Le * np.cos(delta) ** 2)
        return A

    @staticmethod
    def _B_c(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        B = np.zeros((NX, NU))
        B[IV, IA] = 1.0
        B[IDELTA, IDDELTA] = 1.0
        return B

    def _F(self, x: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
        h = self.cfg.dt
        f = self.field
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3, u)
        xn = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[IPSI] = wrap_to_pi(xn[IPSI])
        return xn

    def _F_jac(self, x: np.ndarray, u: np.ndarray, k: int):
        return rk4_jacobians(self._A_c, self._B_c, self.field, x, u, self.cfg.dt)

    # --- cost ----------------------------------------------------------------

    def _residual(self, x: np.ndarray, k: int) -> np.ndarray:
        """Tracking error in the **reference path frame**.

        ``[e_lon, e_lat, e_psi, e_v, delta]``, where the position error is
        rotated into the reference tangent/normal.  The heading residual is
        wrapped: a reference crossing ``+-pi`` otherwise produces a ``2 pi``
        error and one step of full-lock steering.
        """
        ref = self._ref[k]
        c, s_ = self._ref_cos[k], self._ref_sin[k]
        dx, dy = x[IX] - ref[0], x[IY] - ref[1]
        return np.array(
            [
                dx * c + dy * s_,
                -dx * s_ + dy * c,
                wrap_to_pi(x[IPSI] - ref[2]),
                x[IV] - ref[3],
                x[IDELTA],
            ]
        )

    def _residual_jacobian(self, k: int) -> np.ndarray:
        """``dr/dx`` -- a rotation on the position block, identity elsewhere."""
        c, s_ = self._ref_cos[k], self._ref_sin[k]
        J = np.eye(NX)
        J[0, IX], J[0, IY] = c, s_
        J[1, IX], J[1, IY] = -s_, c
        return J

    def _stage_cost(self, x, u, k) -> float:
        Q = np.asarray(self.cfg.Q)
        R = np.asarray(self.cfg.R)
        r = self._residual(x, k)
        return float(r @ (Q * r) + u @ (R * u))

    def _stage_derivs(self, x, u, k):
        Q = np.asarray(self.cfg.Q)
        R = np.asarray(self.cfg.R)
        r = self._residual(x, k)
        J = self._residual_jacobian(k)
        return (
            2 * J.T @ (Q * r),
            2 * R * u,
            2 * J.T @ np.diag(Q) @ J,
            2 * np.diag(R),
            np.zeros((NU, NX)),
        )

    def _terminal_cost(self, x) -> float:
        Qf = np.asarray(self.cfg.Qf)
        r = self._residual(x, self.cfg.horizon)
        return float(r @ (Qf * r))

    def _terminal_derivs(self, x):
        Qf = np.asarray(self.cfg.Qf)
        N = self.cfg.horizon
        J = self._residual_jacobian(N)
        return 2 * J.T @ (Qf * self._residual(x, N)), 2 * J.T @ np.diag(Qf) @ J

    # --- constraints ---------------------------------------------------------

    def _constraints(self, x, u, k) -> np.ndarray:
        cfg = self.cfg
        d_max = cfg.delta_margin * self.p.actuator.delta_max
        a_y = x[IV] ** 2 * np.tan(x[IDELTA]) / self._effective_wheelbase(x[IV])
        out = [
            x[IDELTA] - d_max,
            -x[IDELTA] - d_max,
            x[IV] - cfg.v_max,
            cfg.v_min - x[IV],
            a_y - cfg.a_y_max,
            -a_y - cfg.a_y_max,
        ]
        cor = self._corridor[k]
        if cor is not None:
            e = float((x[:2] - cor.p_ref) @ cor.normal)
            out += [e - cor.hi, cor.lo - e]
        if self._stop is not None:
            p_front = x[:2] + self._front_offset * np.array([np.cos(x[IPSI]), np.sin(x[IPSI])])
            out.append(float((p_front - self._stop.point) @ self._stop.tangent))
        for centre, heading, a, b in self._obs[k]:
            for off in self._ego_off:
                p = x[:2] + (off + self._ego_body_offset) * np.array(
                    [np.cos(x[IPSI]), np.sin(x[IPSI])]
                )
                d = p - centre
                c_, s_ = np.cos(heading), np.sin(heading)
                d_lon, d_lat = d[0] * c_ + d[1] * s_, -d[0] * s_ + d[1] * c_
                out.append(1.0 - float(np.hypot(d_lon / a, d_lat / b)))
        return np.array(out, dtype=float)

    def _constraint_jac(self, x, u, k):
        cfg = self.cfg
        rows_x, rows_u = [], []

        def add(gx):
            rows_x.append(gx)
            rows_u.append(np.zeros(NU))

        add(_unit(NX, IDELTA))
        add(-_unit(NX, IDELTA))
        add(_unit(NX, IV))
        add(-_unit(NX, IV))

        v, delta = x[IV], x[IDELTA]
        Le = self._effective_wheelbase(v)
        g = np.zeros(NX)
        if self.cfg.understeer_correction:
            K = self.p.understeer_gradient
            g[IV] = np.tan(delta) * v * (2 * self.p.L) / Le**2
        else:
            g[IV] = 2 * v * np.tan(delta) / Le
        g[IDELTA] = v**2 / (Le * np.cos(delta) ** 2)
        add(g)
        add(-g)

        cor = self._corridor[k]
        if cor is not None:
            g = np.zeros(NX)
            g[IX], g[IY] = cor.normal
            add(g)
            add(-g)

        psi = x[IPSI]
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        if self._stop is not None:
            tg = self._stop.tangent
            g = np.zeros(NX)
            g[IX], g[IY] = tg
            g[IPSI] = self._front_offset * float(tg @ np.array([-s_psi, c_psi]))
            add(g)
        for centre, heading, a, b in self._obs[k]:
            ch, sh = np.cos(heading), np.sin(heading)
            for off in self._ego_off:
                r = off + self._ego_body_offset
                p = x[:2] + r * np.array([c_psi, s_psi])
                d = p - centre
                d_lon, d_lat = d[0] * ch + d[1] * sh, -d[0] * sh + d[1] * ch
                n = float(np.hypot(d_lon / a, d_lat / b))
                if n < 1e-9:
                    add(np.zeros(NX))
                    continue
                # dn/dp, rotated back into world coordinates
                dn_dframe = np.array([d_lon / a**2, d_lat / b**2]) / n
                dn_dp = np.array(
                    [
                        dn_dframe[0] * ch - dn_dframe[1] * sh,
                        dn_dframe[0] * sh + dn_dframe[1] * ch,
                    ]
                )
                g = np.zeros(NX)
                g[IX], g[IY] = -dn_dp
                g[IPSI] = -float(dn_dp @ (r * np.array([-s_psi, c_psi])))
                add(g)
        return np.array(rows_x), np.array(rows_u)

    # --- context ------------------------------------------------------------

    def _set_context(self, reference, corridor, predictions, stop_line=None):
        cfg = self.cfg
        self._stop = stop_line
        N = cfg.horizon
        self._ref = np.asarray(reference, dtype=float).reshape(N + 1, 4)
        self._ref_cos = np.cos(self._ref[:, 2])
        self._ref_sin = np.sin(self._ref[:, 2])
        self._corridor = list(corridor) if corridor is not None else [None] * (N + 1)
        if len(self._corridor) != N + 1:
            raise ValueError("corridor must have horizon + 1 entries")

        times = np.arange(N + 1) * cfg.dt
        self._obs = [[] for _ in range(N + 1)]
        preds = sorted(
            predictions,
            key=lambda pr: float(np.linalg.norm(pr.positions[0] - self._ref[0, :2])),
        )[: cfg.max_obstacles]
        for pred in preds:
            obj_off, obj_r = multi_circle_cover(pred.length, pred.width, cfg.n_circles)
            sig_lon, sig_lat = pred.ellipse_axes(cfg.inflate_sigma)
            for k, t in enumerate(times):
                if t > pred.times[-1]:
                    continue
                c = pred.position_at(float(t))
                h = float(np.interp(t, pred.times, np.unwrap(pred.headings)))
                a = self._ego_r + obj_r + float(np.interp(t, pred.times, sig_lon))
                b = self._ego_r + obj_r + float(np.interp(t, pred.times, sig_lat))
                for off in obj_off:
                    self._obs[k].append(
                        (c + off * np.array([np.cos(h), np.sin(h)]), h, a, b)
                    )

        # The constraint vector must have the same length at every stage, or the
        # multipliers cannot be indexed consistently.  Pad with inactive rows.
        counts = [len(self._constraints(np.zeros(NX), np.zeros(NU), k)) for k in range(N)]
        self._nc = max(counts) if counts else 0
        self._pad = [self._nc - c for c in counts]

    def _c_padded(self, x, u, k):
        c = self._constraints(x, u, k)
        pad = self._pad[k] if k < len(self._pad) else 0
        return np.concatenate([c, -np.ones(pad)]) if pad else c

    def _cj_padded(self, x, u, k):
        cx, cu = self._constraint_jac(x, u, k)
        pad = self._pad[k] if k < len(self._pad) else 0
        if pad:
            cx = np.vstack([cx, np.zeros((pad, NX))])
            cu = np.vstack([cu, np.zeros((pad, NU))])
        return cx, cu

    # --- solve ---------------------------------------------------------------

    def reset(self) -> None:
        self._U_prev = None
        self._lam_prev = None
        self._penalty_prev = None

    def solve(
        self,
        x0: np.ndarray,
        reference: np.ndarray,
        corridor: list[CorridorStage | None] | None = None,
        predictions: list[Prediction] | None = None,
        stop_line: StopLineConstraint | None = None,
    ) -> MPCResult:
        """One receding-horizon solve.

        ``x0`` is ``[X_r, Y_r, psi, v, delta]`` at the **rear axle**.
        ``reference`` is ``(N+1, 4)`` of ``[X, Y, psi, v]``.
        """
        cfg = self.cfg
        self._set_context(reference, corridor, predictions or [], stop_line)

        act = self.p.actuator
        u_lo = np.array([act.a_min, -act.delta_rate_max])
        u_hi = np.array([act.a_max, act.delta_rate_max])

        solver = ILQR(
            dynamics=self._F,
            stage_cost=self._stage_cost,
            terminal_cost=self._terminal_cost,
            nx=NX,
            nu=NU,
            horizon=cfg.horizon,
            u_lo=u_lo,
            u_hi=u_hi,
            constraints=self._c_padded,
            n_constraints=self._nc,
            options=ILQROptions(
                max_iter=cfg.max_iter,
                max_al_iter=cfg.max_al_iter,
                time_budget=cfg.time_budget,
            ),
            cost_derivatives=self._stage_derivs,
            terminal_derivatives=self._terminal_derivs,
            constraint_jacobians=self._cj_padded,
            dynamics_jacobians=self._F_jac,
        )

        U0 = None
        lam0 = None
        if self._U_prev is not None and self._U_prev.shape == (cfg.horizon, NU):
            U0 = np.vstack([self._U_prev[1:], self._U_prev[-1:]])
            if self._lam_prev is not None and self._lam_prev.shape == (cfg.horizon, self._nc):
                lam0 = np.vstack([self._lam_prev[1:], self._lam_prev[-1:]])

        res = solver.solve(np.asarray(x0, dtype=float), U0, lam0, self._penalty_prev)
        self._U_prev = res.U.copy()
        self._lam_prev = res.lam.copy() if res.lam is not None else None
        self._penalty_prev = res.penalty

        Le = np.array([self._effective_wheelbase(v) for v in res.X[:, IV]])
        a_y = res.X[:, IV] ** 2 * np.tan(res.X[:, IDELTA]) / Le
        return MPCResult(
            a_cmd=float(res.U[0, IA]),
            delta_cmd=float(res.X[1, IDELTA]),
            X=res.X,
            U=res.U,
            cost=res.cost,
            violation=res.constraint_violation,
            status=res.status,
            solve_time=res.solve_time,
            iterations=res.iterations,
            feasible=res.constraint_violation <= 5e-2,
            predicted_a_y=a_y,
        )


def _unit(n: int, i: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v
