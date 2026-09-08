"""Constrained iLQR / DDP with an augmented-Lagrangian outer loop.

This is the nonlinear solver behind :mod:`avsim.control.mpc`.  It solves

    min  sum_k l(x_k, u_k, k) + l_f(x_N)
    s.t. x_{k+1} = F(x_k, u_k, k)          (satisfied by construction)
         u_lo <= u_k <= u_hi               (enforced in the backward pass)
         c_i(x_k, u_k, k) <= 0             (augmented Lagrangian)

Three design choices, each with a reason:

**Derivatives of the discrete map.**  ``F`` is the *discrete* dynamics -- the
RK4 step the simulator would take -- and its Jacobians are taken of that map,
not of the continuous field.  Lecture 2/3: "inside a nonlinear optimizer, use
derivatives of the discrete constraint you actually evaluate."

**Box constraints in the backward pass, not the cost.**  Input limits are
enforced by clamping the control update and zeroing the corresponding rows of
the feedback gain (Tassa's control-limited DDP).  Penalizing them instead
produces a controller that exceeds the limit whenever the tracking error is
large enough to outweigh the penalty -- which is exactly when it must not.

**Augmented Lagrangian for the rest.**  A pure penalty needs an infinite weight
for an exactly-satisfied constraint and wrecks the conditioning long before it
gets there.  The multiplier update lets a moderate penalty converge to the true
constrained optimum.

Rollouts are *feasible by construction*: iLQR is a single-shooting method, so
the dynamics are never violated, only the path constraints are.  That is the
trade named in the lecture's transcription slide -- fewer variables, but a
horizon-long composition -- and it is chosen here because the horizons are
short (2-4 s) and a shooting method needs no QP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Dynamics = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
StageCost = Callable[[np.ndarray, np.ndarray, int], float]
TerminalCost = Callable[[np.ndarray], float]
Constraint = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class ILQRResult:
    """Everything the caller needs, including why it stopped."""

    X: np.ndarray                #: (N+1, nx) state trajectory
    U: np.ndarray                #: (N, nu) input trajectory
    K: np.ndarray                #: (N, nu, nx) feedback gains
    k_ff: np.ndarray             #: (N, nu) feedforward updates
    cost: float
    constraint_violation: float
    iterations: int
    converged: bool
    status: str
    solve_time: float = 0.0
    cost_history: list[float] = field(default_factory=list)
    #: multipliers and penalty at exit, for warm-starting the next solve
    lam: np.ndarray | None = None
    penalty: float = 0.0


@dataclass
class ILQROptions:
    max_iter: int = 40
    max_al_iter: int = 6
    tol_cost: float = 1e-5
    tol_violation: float = 1e-3
    #: Levenberg-Marquardt regularization on the state Hessian
    reg_init: float = 1e-6
    reg_min: float = 1e-8
    reg_max: float = 1e8
    reg_factor: float = 4.0
    #: augmented-Lagrangian penalty schedule
    penalty_init: float = 10.0
    penalty_factor: float = 6.0
    penalty_max: float = 1e6
    line_search: tuple[float, ...] = (1.0, 0.7, 0.45, 0.28, 0.15, 0.08, 0.03, 0.01)
    fd_eps: float = 1e-6
    #: Wall-clock budget [s]; ``None`` means run to convergence.  A real
    #: controller has a deadline, and a solver that silently overruns it has
    #: not solved the control problem -- it has solved a different one, late.
    time_budget: float | None = None


class ILQR:
    """Iterative LQR with control box limits and augmented-Lagrangian constraints."""

    def __init__(
        self,
        dynamics: Dynamics,
        stage_cost: StageCost,
        terminal_cost: TerminalCost,
        nx: int,
        nu: int,
        horizon: int,
        u_lo: np.ndarray | None = None,
        u_hi: np.ndarray | None = None,
        constraints: Constraint | None = None,
        n_constraints: int = 0,
        options: ILQROptions | None = None,
        cost_derivatives: Callable | None = None,
        constraint_jacobians: Callable | None = None,
        terminal_derivatives: Callable | None = None,
        dynamics_jacobians: Callable | None = None,
    ):
        """
        ``cost_derivatives(x, u, k) -> (lx, lu, lxx, luu, lux)`` and
        ``constraint_jacobians(x, u, k) -> (cx, cu)`` are optional analytic
        hooks.  They are worth supplying: with finite-difference Hessians the
        solver needs ``O(nx^2)`` cost evaluations per stage per iteration, which
        for a 5-state problem over a 40-step horizon is a few hundred thousand
        evaluations and nowhere near real time.  With the analytic quadratic
        cost and a Gauss-Newton treatment of the augmented-Lagrangian terms the
        same problem solves in milliseconds.
        """
        self.F = dynamics
        self.l = stage_cost
        self.lf = terminal_cost
        self.nx, self.nu, self.N = int(nx), int(nu), int(horizon)
        self.u_lo = -np.inf * np.ones(nu) if u_lo is None else np.asarray(u_lo, float)
        self.u_hi = np.inf * np.ones(nu) if u_hi is None else np.asarray(u_hi, float)
        self.c = constraints
        self.nc = int(n_constraints)
        self.opt = options or ILQROptions()

        self.cost_derivatives = cost_derivatives
        self.constraint_jacobians = constraint_jacobians
        self.terminal_derivatives = terminal_derivatives
        self.dynamics_jacobians = dynamics_jacobians
        self._project_hessians = cost_derivatives is None
        self._eye_x = np.eye(self.nx)
        self._eye_u = np.eye(self.nu)

        self.lam = np.zeros((self.N, self.nc)) if self.nc else np.zeros((self.N, 0))
        self.penalty = self.opt.penalty_init

    # --- derivatives ---------------------------------------------------------

    def _dyn_jacobians(self, x: np.ndarray, u: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.dynamics_jacobians is not None:
            return self.dynamics_jacobians(x, u, k)
        eps = self.opt.fd_eps
        A = np.empty((self.nx, self.nx))
        B = np.empty((self.nx, self.nu))
        for i in range(self.nx):
            d = np.zeros(self.nx)
            d[i] = eps * max(1.0, abs(x[i]))
            A[:, i] = (self.F(x + d, u, k) - self.F(x - d, u, k)) / (2 * d[i])
        for j in range(self.nu):
            d = np.zeros(self.nu)
            d[j] = eps * max(1.0, abs(u[j]))
            B[:, j] = (self.F(x, u + d, k) - self.F(x, u - d, k)) / (2 * d[j])
        return A, B

    def _constraint_jacobians(self, x, u, k):
        if self.constraint_jacobians is not None:
            return self.constraint_jacobians(x, u, k)
        eps = 1e-6
        cx = np.empty((self.nc, self.nx))
        cu = np.empty((self.nc, self.nu))
        for i in range(self.nx):
            d = _e(self.nx, i, eps)
            cx[:, i] = (np.asarray(self.c(x + d, u, k)) - np.asarray(self.c(x - d, u, k))) / (2 * eps)
        for j in range(self.nu):
            d = _e(self.nu, j, eps)
            cu[:, j] = (np.asarray(self.c(x, u + d, k)) - np.asarray(self.c(x, u - d, k))) / (2 * eps)
        return cx, cu

    def _base_cost_derivatives(self, x, u, k):
        if self.cost_derivatives is not None:
            return self.cost_derivatives(x, u, k)
        eps = 1e-5
        L = self.l
        lx = np.array([(L(x + _e(self.nx, i, eps), u, k) - L(x - _e(self.nx, i, eps), u, k)) / (2 * eps) for i in range(self.nx)])
        lu = np.array([(L(x, u + _e(self.nu, j, eps), k) - L(x, u - _e(self.nu, j, eps), k)) / (2 * eps) for j in range(self.nu)])
        lxx = _fd_hessian(lambda xx: L(xx, u, k), x, eps)
        luu = _fd_hessian(lambda uu: L(x, uu, k), u, eps)
        lux = np.zeros((self.nu, self.nx))
        for j in range(self.nu):
            du = _e(self.nu, j, eps)
            gp = np.array([(L(x + _e(self.nx, i, eps), u + du, k) - L(x - _e(self.nx, i, eps), u + du, k)) / (2 * eps) for i in range(self.nx)])
            gm = np.array([(L(x + _e(self.nx, i, eps), u - du, k) - L(x - _e(self.nx, i, eps), u - du, k)) / (2 * eps) for i in range(self.nx)])
            lux[j] = (gp - gm) / (2 * eps)
        return lx, lu, lxx, luu, lux

    def _cost_derivatives(self, x: np.ndarray, u: np.ndarray, k: int):
        """Derivatives of the augmented stage cost.

        The base cost contributes exactly; the augmented-Lagrangian terms
        contribute their gradient exactly and a **Gauss-Newton** Hessian
        ``mu * c_z^T I c_z``, dropping the ``c_zz`` term.  Gauss-Newton is
        positive semidefinite by construction, which is what keeps the backward
        pass solvable near an active constraint where the exact Hessian is
        routinely indefinite.
        """
        lx, lu, lxx, luu, lux = self._base_cost_derivatives(x, u, k)
        lx, lu = np.asarray(lx, float).copy(), np.asarray(lu, float).copy()
        lxx, luu, lux = np.asarray(lxx, float).copy(), np.asarray(luu, float).copy(), np.asarray(lux, float).copy()

        if self.nc:
            c = np.asarray(self.c(x, u, k), dtype=float)
            lam = self.lam[k]
            active = (c > 0.0) | (lam > 0.0)
            cx, cu = self._constraint_jacobians(x, u, k)
            w = lam + self.penalty * np.where(active, c, 0.0)
            lx += cx.T @ w
            lu += cu.T @ w
            I_mu = self.penalty * np.diag(active.astype(float))
            lxx += cx.T @ I_mu @ cx
            luu += cu.T @ I_mu @ cu
            lux += cu.T @ I_mu @ cx

        # Analytic quadratic costs are positive semidefinite by construction and
        # the Gauss-Newton augmented terms add PSD blocks, so the projection is
        # only needed when the base Hessians came from finite differences.
        if self._project_hessians:
            lxx, luu = _psd(lxx), _psd(luu)
        return lx, lu, lxx, luu, lux

    def _terminal_derivatives(self, x: np.ndarray):
        if self.terminal_derivatives is not None:
            return self.terminal_derivatives(x)
        eps = 1e-5
        vx = np.empty(self.nx)
        for i in range(self.nx):
            d = _e(self.nx, i, eps)
            vx[i] = (self.lf(x + d) - self.lf(x - d)) / (2 * eps)
        return vx, _psd(_fd_hessian(self.lf, x, eps))

    def warm_start(self, U_prev: np.ndarray, shift: int = 1) -> np.ndarray:
        """Shift a previous input sequence forward, repeating the last entry.

        The standard MPC warm start.  It matters more than it looks: from a
        shifted solution the solver typically converges in one or two
        iterations, which is the difference between a controller that runs at
        the sample rate and one that does not.
        """
        U = np.asarray(U_prev, dtype=float).reshape(self.N, self.nu)
        return np.vstack([U[shift:], np.repeat(U[-1:], shift, axis=0)])

    # --- augmented cost ------------------------------------------------------

    def _augmented_cost(self, x: np.ndarray, u: np.ndarray, k: int) -> float:
        cost = float(self.l(x, u, k))
        if self.nc:
            c = np.asarray(self.c(x, u, k), dtype=float)
            lam = self.lam[k]
            # Active set: a constraint is penalized when it is violated, or when
            # its multiplier is still pushing.
            active = (c > 0.0) | (lam > 0.0)
            cost += float(lam @ c + 0.5 * self.penalty * np.sum(np.where(active, c, 0.0) ** 2))
        return cost

    def _violation(self, X: np.ndarray, U: np.ndarray) -> float:
        if not self.nc:
            return 0.0
        worst = 0.0
        for k in range(self.N):
            c = np.asarray(self.c(X[k], U[k], k), dtype=float)
            worst = max(worst, float(np.max(np.maximum(c, 0.0))) if c.size else 0.0)
        return worst

    # --- passes --------------------------------------------------------------

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        X = np.empty((self.N + 1, self.nx))
        X[0] = x0
        for k in range(self.N):
            X[k + 1] = self.F(X[k], U[k], k)
        return X

    def total_cost(self, X: np.ndarray, U: np.ndarray) -> float:
        return float(sum(self._augmented_cost(X[k], U[k], k) for k in range(self.N)) + self.lf(X[self.N]))

    def _backward(self, X: np.ndarray, U: np.ndarray, reg: float):
        N, nx, nu = self.N, self.nx, self.nu
        K = np.zeros((N, nu, nx))
        kff = np.zeros((N, nu))
        Vx, Vxx = self._terminal_derivatives(X[N])
        dV = np.zeros(2)

        for k in range(N - 1, -1, -1):
            A, B = self._dyn_jacobians(X[k], U[k], k)
            lx, lu, lxx, luu, lux = self._cost_derivatives(X[k], U[k], k)

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Vxx_reg = Vxx + reg * self._eye_x
            Quu = luu + B.T @ Vxx_reg @ B
            Qux = lux + B.T @ Vxx_reg @ A

            Quu = 0.5 * (Quu + Quu.T) + 1e-9 * self._eye_u
            try:
                np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError:
                # Not positive definite: signal the caller to raise the
                # Levenberg-Marquardt regularization and try again.
                return None

            # Control-limited update: solve the unconstrained step, clamp it to
            # the box, and zero the gain rows of the clamped channels so the
            # feedback cannot push the input back outside during the forward pass.
            k_un = -np.linalg.solve(Quu, Qu)
            k_cl = np.clip(U[k] + k_un, self.u_lo, self.u_hi) - U[k]
            free = np.abs(k_cl - k_un) < 1e-9

            K_k = np.zeros((nu, nx))
            if free.any():
                idx = np.where(free)[0]
                Quu_ff = Quu[np.ix_(idx, idx)]
                K_k[idx] = -np.linalg.solve(Quu_ff, Qux[idx])
            kff[k] = k_cl
            K[k] = K_k

            dV += np.array([k_cl @ Qu, 0.5 * k_cl @ Quu @ k_cl])
            Vx = Qx + K_k.T @ Quu @ k_cl + K_k.T @ Qu + Qux.T @ k_cl
            Vxx = Qxx + K_k.T @ Quu @ K_k + K_k.T @ Qux + Qux.T @ K_k
            Vxx = 0.5 * (Vxx + Vxx.T)
        return K, kff, dV

    def _forward(self, x0, X, U, K, kff, alpha):
        Xn = np.empty_like(X)
        Un = np.empty_like(U)
        Xn[0] = x0
        for k in range(self.N):
            du = alpha * kff[k] + K[k] @ (Xn[k] - X[k])
            Un[k] = np.clip(U[k] + du, self.u_lo, self.u_hi)
            Xn[k + 1] = self.F(Xn[k], Un[k], k)
        return Xn, Un

    # --- driver --------------------------------------------------------------

    def solve(
        self,
        x0: np.ndarray,
        U_init: np.ndarray | None = None,
        lam_init: np.ndarray | None = None,
        penalty_init: float | None = None,
    ) -> ILQRResult:
        """Solve from ``x0``, optionally warm-starting inputs *and* multipliers.

        Warm-starting the multipliers matters as much as warm-starting the
        inputs: restarting the augmented Lagrangian from zero every step makes
        the solver rediscover which constraints are active, and the receding
        horizon then costs more per step than a cold solve, not less.
        """
        import time

        t_start = time.perf_counter()
        x0 = np.asarray(x0, dtype=float)
        U = (
            np.zeros((self.N, self.nu))
            if U_init is None
            else np.clip(np.asarray(U_init, dtype=float).reshape(self.N, self.nu), self.u_lo, self.u_hi)
        )
        self.lam = (
            np.zeros((self.N, self.nc))
            if lam_init is None
            else np.asarray(lam_init, dtype=float).reshape(self.N, self.nc).copy()
        )
        self.penalty = self.opt.penalty_init if penalty_init is None else float(penalty_init)

        X = self.rollout(x0, U)
        cost = self.total_cost(X, U)
        history = [cost]
        reg = self.opt.reg_init
        iters = 0
        status = "max_al_iter"

        for _ in range(max(self.opt.max_al_iter, 1)):
            for _ in range(self.opt.max_iter):
                if self.opt.time_budget is not None and time.perf_counter() - t_start > self.opt.time_budget:
                    status = "time_budget"
                    break
                iters += 1
                bp = self._backward(X, U, reg)
                if bp is None:
                    reg = min(reg * self.opt.reg_factor, self.opt.reg_max)
                    if reg >= self.opt.reg_max:
                        status = "regularization_limit"
                        break
                    continue
                K, kff, dV = bp

                improved = False
                for alpha in self.opt.line_search:
                    Xn, Un = self._forward(x0, X, U, K, kff, alpha)
                    new_cost = self.total_cost(Xn, Un)
                    expected = -(alpha * dV[0] + alpha**2 * dV[1])
                    ratio = (cost - new_cost) / expected if abs(expected) > 1e-12 else (1.0 if new_cost < cost else -1.0)
                    if new_cost < cost and ratio > 1e-4:
                        X, U = Xn, Un
                        improved = True
                        break
                if not improved:
                    reg = min(reg * self.opt.reg_factor, self.opt.reg_max)
                    if reg >= self.opt.reg_max:
                        status = "line_search_failed"
                        break
                    continue

                reg = max(reg / self.opt.reg_factor, self.opt.reg_min)
                delta = cost - new_cost
                cost = new_cost
                history.append(cost)
                if delta < self.opt.tol_cost * max(1.0, abs(cost)):
                    status = "converged"
                    break

            if status == "time_budget":
                break
            viol = self._violation(X, U)
            if viol <= self.opt.tol_violation:
                status = "converged" if status in ("converged", "max_al_iter") else status
                break
            if not self.nc:
                break
            # Multiplier and penalty update.
            for k in range(self.N):
                c = np.asarray(self.c(X[k], U[k], k), dtype=float)
                self.lam[k] = np.maximum(0.0, self.lam[k] + self.penalty * c)
            self.penalty = min(self.penalty * self.opt.penalty_factor, self.opt.penalty_max)
            cost = self.total_cost(X, U)

        viol = self._violation(X, U)
        return ILQRResult(
            X=X,
            U=U,
            K=K if "K" in dir() else np.zeros((self.N, self.nu, self.nx)),
            k_ff=kff if "kff" in dir() else np.zeros((self.N, self.nu)),
            cost=float(cost),
            constraint_violation=float(viol),
            iterations=iters,
            converged=viol <= self.opt.tol_violation and status == "converged",
            status=status,
            solve_time=time.perf_counter() - t_start,
            cost_history=history,
            lam=self.lam.copy(),
            penalty=self.penalty,
        )


# --- small helpers ------------------------------------------------------------

def _e(n: int, i: int, eps: float) -> np.ndarray:
    d = np.zeros(n)
    d[i] = eps
    return d


def _fd_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = _e(n, i, eps)
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / eps**2
        for j in range(i + 1, n):
            ej = _e(n, j, eps)
            H[i, j] = H[j, i] = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4 * eps**2)
    return H


def _psd(H: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the positive semidefinite cone.

    Cholesky first: for an analytic quadratic cost the matrix is already
    positive definite and the factorization succeeds, which costs a fraction of
    an eigendecomposition.  The eigenvalue clip is the fallback for the cases
    that actually need it -- an indefinite finite-difference Hessian, or an
    exact Hessian near an active constraint.
    """
    H = 0.5 * (H + H.T)
    try:
        np.linalg.cholesky(H + 1e-12 * np.eye(len(H)))
        return H
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(H)
        return (V * np.maximum(w, 0.0)) @ V.T
