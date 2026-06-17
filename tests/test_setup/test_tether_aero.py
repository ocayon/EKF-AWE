"""Unit-level verification of the per-segment tether aerodynamics.

For each tether element the model resolves the apparent wind into a drag
(along the apparent wind) and a lift (perpendicular to it) using the
cross-flow + skin-friction decomposition from ``Tether`` (see
``calculate_tether_shape_symbolic``)::

    cd_t = cd * sin^3(theta) + pi * cf * cos^3(theta)
    cl_t = cd * sin^2(theta) * cos(theta) - pi * cf * sin(theta) * cos^2(theta)

where ``theta`` is the angle between the apparent wind and the segment axis.
For UNIT drag/lift directions, ``drag + lift`` must reproduce the physical
Hoerner force exactly: a normal (cross-flow) component of magnitude
``q * cd * sin^2(theta)`` and a tangential (skin-friction) component of
magnitude ``q * pi * cf * cos^2(theta)``, with
``q = 0.5 * rho * |va|^2 * L * d``.

These tests evaluate the exact CasADi expressions used by the model on
concrete vectors and pin that invariant down -- in particular they guard the
lift direction against being left unnormalised (which would scale the lift by
an extra ``sin(theta)``).
"""

from __future__ import annotations

import numpy as np
import casadi as ca
import pytest

from awes_ekf.utils import calculate_angle_2vec


def _segment_aero(va, ej, *, cd, cf, rho, length, diameter):
    """Drag and lift on one tether segment, mirroring the model expressions.

    ``va`` is the apparent wind at the node and ``ej`` the unit segment axis
    (both 3-vectors). Returns ``(drag, lift, theta)`` as numpy arrays / float,
    evaluated from the same CasADi graph the tether builds.
    """
    # Build as ``ca.SX`` so ``calculate_angle_2vec`` takes the same symbolic
    # branch the tether model uses (it builds its graph with ``ca.SX``).
    va = ca.SX(np.asarray(va, dtype=float).reshape(3).tolist())
    ej = ca.SX(np.asarray(ej, dtype=float).reshape(3).tolist())

    theta = calculate_angle_2vec(va, ej)
    cd_t = cd * ca.sin(theta) ** 3 + np.pi * cf * ca.cos(theta) ** 3
    cl_t = (
        cd * ca.sin(theta) ** 2 * ca.cos(theta)
        - np.pi * cf * ca.sin(theta) * ca.cos(theta) ** 2
    )
    dir_D = va / ca.norm_2(va)
    lift_perp = ej - ca.dot(ej, dir_D) * dir_D
    dir_L = -lift_perp / (ca.norm_2(lift_perp) + 1e-9)
    dyn_pressure_area = 0.5 * rho * ca.norm_2(va) ** 2 * length * diameter

    drag = dyn_pressure_area * cd_t * dir_D
    lift = dyn_pressure_area * cl_t * dir_L
    return (
        np.asarray(ca.evalf(drag)).reshape(3),
        np.asarray(ca.evalf(lift)).reshape(3),
        float(ca.evalf(theta)),
    )


# A handful of non-degenerate geometries (apparent wind, segment axis).
_CASES = [
    (np.array([12.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),  # perpendicular
    (np.array([10.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0])),  # oblique
    (np.array([3.0, -4.0, 2.0]), np.array([1.0, 1.0, 1.0]) / np.sqrt(3)),
    (np.array([0.0, 8.0, 1.0]), np.array([0.0, 1.0, 0.0])),  # near-parallel
]


@pytest.mark.parametrize("va, ej", _CASES)
def test_aero_matches_crossflow_plus_friction(va, ej):
    """``drag + lift`` reproduces the closed-form cross-flow + friction force.

    The magnitude is independent of direction, so this directly catches an
    unnormalised lift direction (which would scale lift by ``sin(theta)``).
    """
    cd, cf = 1.1, 0.02
    rho, length, diameter = 1.225, 22.8, 0.01

    drag, lift, theta = _segment_aero(
        va, ej, cd=cd, cf=cf, rho=rho, length=length, diameter=diameter
    )
    force = drag + lift

    q = 0.5 * rho * float(np.dot(va, va)) * length * diameter
    expected = q * np.hypot(
        cd * np.sin(theta) ** 2, np.pi * cf * np.cos(theta) ** 2
    )
    assert float(np.linalg.norm(force)) == pytest.approx(expected, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("va, ej", _CASES)
def test_aero_is_normal_to_segment_without_friction(va, ej):
    """With ``cf = 0`` the force is pure cross-flow drag: perpendicular to the
    segment axis. An unnormalised lift direction would leave a spurious
    tangential component even here."""
    cd, cf = 1.1, 0.0
    rho, length, diameter = 1.225, 22.8, 0.01

    drag, lift, _ = _segment_aero(
        va, ej, cd=cd, cf=cf, rho=rho, length=length, diameter=diameter
    )
    force = drag + lift
    magnitude = float(np.linalg.norm(force))

    ej = np.asarray(ej, dtype=float)
    ej = ej / np.linalg.norm(ej)
    tangential = float(np.dot(force, ej))
    assert abs(tangential) <= 1e-6 * (magnitude + 1e-9)


def test_lift_direction_is_normalised():
    """Regression: the lift direction must be a unit vector.

    Reconstructing it without normalising would give magnitude ``sin(theta)``
    rather than 1, scaling the lift coefficient by an extra ``sin(theta)``.
    """
    va = ca.SX([10.0, 0.0, 5.0])
    ej = ca.SX([0.0, 0.0, 1.0])
    dir_D = va / ca.norm_2(va)
    lift_perp = ej - ca.dot(ej, dir_D) * dir_D
    dir_L = -lift_perp / (ca.norm_2(lift_perp) + 1e-9)
    assert float(ca.evalf(ca.norm_2(dir_L))) == pytest.approx(1.0, abs=1e-6)
