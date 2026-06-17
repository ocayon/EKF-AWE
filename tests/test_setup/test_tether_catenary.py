"""Analytical catenary verification for the lumped-mass ``Tether``.

When the tether hangs under its own weight alone -- no wind, no aerodynamic
drag, no kite rotation -- its shape is an inextensible catenary. This test
pins down a catenary by choosing the parameter ``a = H/w`` (horizontal
tension over weight-per-length) and the horizontal span ``x_k``, derives the
matching kite position, total length and ground tension analytically, hands
those into ``Tether``, and checks that the ground->kite solver recovers the
same shape.

``Tether`` iterates from the ground anchor up to the kite, so the natural
configuration places the catenary low point AT the ground anchor
(``x_b = 0``), which makes the ground-side tangent horizontal. This gives the
closed form::

    z(x)        = a * (cosh(x/a) - 1)
    s(x)        = a * sinh(x/a)               (arc length from the ground)
    H           = w * a                        (horizontal tension, constant)
    elevation_0 = 0                            (ground tangent is horizontal)

with the kite at ``x = x_k`` and weight-per-length ``w = rho_t * A * g``.

The module-level air density ``rho`` is patched to zero so the aerodynamic
term vanishes and the only load on the tether is gravity.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

import awes_ekf.setup.tether as tether_mod
from awes_ekf.setup.tether import Tether
from awes_ekf.setup.settings import ObservationData


class _KiteStub:
    """Minimal kite the tether reads (``mass`` and ``area`` only).

    A massless kite makes the boundary node carry only half a tether segment,
    which is exactly what the analytical catenary sampling below assumes.
    """

    mass = 0.0
    area = 10.0


@pytest.fixture
def no_aero(monkeypatch):
    """Disable the aerodynamic term by zeroing the module-level air density.

    ``Tether`` reads ``rho`` at construction time (the CasADi graph is built
    in ``__init__``), so this must be patched before the tether is created.
    """
    monkeypatch.setattr(tether_mod, "rho", 0.0)


@pytest.fixture(scope="module")
def catenary():
    """Closed-form hanging-tether catenary with low point at the ground."""
    a = 80.0
    x_k = 100.0
    diameter = 0.01
    density = 970.0  # Dyneema-SK78
    g = 9.81

    w = np.pi * diameter**2 / 4.0 * density * g
    z_k = a * (np.cosh(x_k / a) - 1.0)
    length = a * np.sinh(x_k / a)
    horizontal_tension = w * a

    return {
        "a": a,
        "x_k": x_k,
        "z_k": z_k,
        "length": length,
        "horizontal_tension": horizontal_tension,
        "diameter": diameter,
        "density": density,
        "g": g,
        "w": w,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tether(catenary, n_elements, *, elastic=False):
    return Tether(
        _KiteStub(),
        None,  # no KCU
        ObservationData(),  # all optional measurements off -> no extra args
        elastic=elastic,
        material_name="Dyneema-SK78",
        diameter=catenary["diameter"],
        n_elements=n_elements,
    )


def _solve(tether, catenary):
    """Solve the ground->kite shape for the analytic catenary boundary.

    The tether owns ``(elevation_0, azimuth_0, tether_length)`` as unknowns;
    the ground tension magnitude is supplied as a parameter. With the low
    point at the ground the magnitude is exactly the horizontal tension.
    """
    r_kite = np.array([catenary["x_k"], 0.0, catenary["z_k"]])
    v_kite = np.zeros(3)
    # A tiny non-zero wind avoids the 0/0 wind-direction unit vector; with
    # ``rho = 0`` it contributes no force.
    vw = np.array([1e-6, 0.0, 0.0])
    tension_ground = catenary["horizontal_tension"]
    args = (tension_ground, r_kite, v_kite, vw)

    # Straight-line initial guess, slightly stretched so the solver has room
    # to grow the length toward the curved catenary.
    distance = float(np.linalg.norm(r_kite))
    x0 = [0.3, 0.0, distance * 1.05]

    sol = least_squares(
        tether.objective_function,
        x0,
        args=args,
        method="lm",
        xtol=1e-12,
        ftol=1e-12,
    )
    full_args = (sol.x[0], sol.x[1], sol.x[2]) + args
    return sol, full_args


def _analytical_node_positions(catenary, n_elements):
    """Catenary node positions sampled at the equal arc lengths the model uses."""
    a = catenary["a"]
    length = catenary["length"]
    arc = np.linspace(0.0, length, n_elements + 1)
    x = a * np.arcsinh(arc / a)
    z = a * (np.cosh(x / a) - 1.0)
    return np.column_stack([x, np.zeros_like(x), z])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_elastic_defines_axial_stiffness(no_aero, catenary):
    """``Tether(elastic=True)`` defines ``EA = E*area`` and stretches the cable.

    Under tension an elastic tether's stretched length must exceed the
    unstrained length handed in.
    """
    tether = _make_tether(catenary, n_elements=50, elastic=True)
    assert tether.EA == pytest.approx(tether.Youngs_modulus * tether.area)

    unstrained_length = 200.0
    args = (
        0.01,  # elevation_0
        0.0,  # azimuth_0
        unstrained_length,
        5.0e4,  # ground tension -> appreciable stretch
        np.array([100.0, 0.0, 150.0]),  # r_kite
        np.zeros(3),  # v_kite
        np.array([1e-6, 0.0, 0.0]),  # vw
    )
    stretched = float(np.asarray(tether.tether_length(*args)).reshape(-1)[0])
    assert stretched > unstrained_length


def test_recovers_catenary_length_and_ground_angle(no_aero, catenary):
    """Solver recovers the analytic length, horizontal ground tangent, azimuth."""
    tether = _make_tether(catenary, n_elements=200)
    sol, _ = _solve(tether, catenary)

    assert sol.success, sol.message
    elevation_0, azimuth_0, length = sol.x.tolist()

    assert np.max(np.abs(sol.fun)) < 1e-6  # kite boundary residual closed
    assert length == pytest.approx(catenary["length"], rel=5e-3)
    # The low point sits at the anchor, so the ground element is horizontal.
    assert elevation_0 == pytest.approx(0.0, abs=5e-3)
    assert azimuth_0 == pytest.approx(0.0, abs=1e-8)


def test_shape_matches_analytical_catenary(no_aero, catenary):
    """Pointwise positions of the discrete tether match the closed-form curve."""
    n_elements = 200
    tether = _make_tether(catenary, n_elements=n_elements)
    sol, full_args = _solve(tether, catenary)
    assert sol.success, sol.message

    positions = np.asarray(tether.positions(*full_args))
    expected = _analytical_node_positions(catenary, n_elements)
    max_err = float(np.linalg.norm(positions - expected, axis=1).max())

    # 1% of the tether length is comfortably above the O(1/N) discretization
    # error from the lumped-mass model.
    assert max_err < 0.01 * catenary["length"], (
        f"max position error {max_err:.4f} m exceeds 1% of length "
        f"{catenary['length']:.4f} m"
    )


def test_converges_toward_catenary_as_n_increases(no_aero, catenary):
    """Refining the discretization brings the discrete shape closer to the
    closed-form catenary monotonically."""
    errors = []
    for n in (20, 80, 320):
        tether = _make_tether(catenary, n_elements=n)
        sol, full_args = _solve(tether, catenary)
        assert sol.success, sol.message
        positions = np.asarray(tether.positions(*full_args))
        expected = _analytical_node_positions(catenary, n)
        errors.append(float(np.linalg.norm(positions - expected, axis=1).max()))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
