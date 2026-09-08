"""Sign, frame and unit conventions used by every module in :mod:`avsim`.

The whole point of collecting them here is the caution raised in Lecture 2/3:

    "Half the sign errors in this field come from mixing two conventions in
    one file."

Every model, controller and estimator in this package is written against the
single convention documented below.  If you add a module, import this file and
state which reference point you use.

ISO 8855 / SAE J670 (ISO orientation)
-------------------------------------
* ``x`` points forward, ``y`` points to the **left**, ``z`` points up.
* Yaw ``psi`` is measured counter-clockwise from the world ``X`` axis.
* Steering angle ``delta`` is **positive to the left** (positive yaw rate).
* Body-frame origin is the **centre of gravity (CG)** unless a symbol is
  explicitly named ``*_r`` / documented as "rear-axle frame".

Tire slip
---------
* ``alpha = (wheel heading) - (velocity direction)``.
* Therefore ``F_y = C_alpha * alpha`` with ``C_alpha > 0``.
  Texts that define ``alpha`` with the opposite sign write ``F_y = -C_alpha
  alpha``; both are valid, mixing them is not.

Longitudinal slip
-----------------
``kappa_x = (R_w * omega_w - v_x) / max(|v_x|, |R_w omega_w|, eps)``

State ordering
--------------
Two state vectors appear throughout, and they are *not* interchangeable:

``KINEMATIC_REAR_AXLE``  ``x = [X_r, Y_r, psi, v]``      ``u = [a, delta]``
``DYNAMIC_CG``           ``z = [X, Y, psi, v_x, v_y, r]`` ``u = [a, delta]``

The conversion between them requires the lever-arm transform in
:func:`avsim.models.kinematic_bicycle.rear_axle_to_cg`; never mix a state
equation written at one reference point with constraints measured at another.

Units
-----
SI throughout: metres, seconds, radians, kilograms, newtons.  Angles are
radians everywhere; degrees only ever appear in log/plot labels.
"""

from __future__ import annotations

import numpy as np

# --- state layouts -----------------------------------------------------------

#: index map for the rear-axle kinematic bicycle state ``[X_r, Y_r, psi, v]``
KIN_X, KIN_Y, KIN_PSI, KIN_V = 0, 1, 2, 3
KIN_NX = 4

#: index map for the CG dynamic state ``[X, Y, psi, v_x, v_y, r]``
DYN_X, DYN_Y, DYN_PSI, DYN_VX, DYN_VY, DYN_R = 0, 1, 2, 3, 4, 5
DYN_NX = 6

#: index map for the common input ``[a, delta]``
U_A, U_DELTA = 0, 1
NU = 2

GRAVITY = 9.80665  # m/s^2


def wrap_to_pi(angle):
    """Wrap an angle (or array of angles) to ``[-pi, pi)``.

    Used everywhere a heading difference is formed.  Doing this in one place
    keeps the branch cut consistent between the simulator, the estimator and
    the optimizer -- a mismatch there shows up as a controller that briefly
    commands full lock when the vehicle crosses ``+-pi``.
    """
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def sinc_taylor(x):
    """``sin(x)/x`` evaluated without cancellation near zero.

    The SE(2) exponential and its right Jacobian both contain ``sin(t)/t`` and
    ``(1-cos t)/t``.  Evaluating them naively loses all significant digits for
    ``|t| < 1e-4``; the Taylor branch keeps full double precision.
    """
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-6
    out = np.empty_like(x)
    xs = np.where(small, 0.0, x)
    out = np.where(small, 1.0 - x**2 / 6.0 + x**4 / 120.0, np.sin(xs) / np.where(small, 1.0, xs))
    return out


def one_minus_cos_over_x(x):
    """``(1 - cos x) / x`` evaluated without cancellation near zero."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    return np.where(small, x / 2.0 - x**3 / 24.0, (1.0 - np.cos(xs)) / xs)
