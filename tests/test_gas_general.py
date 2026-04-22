"""Tests for jet_ism.gas.general — physical calculations on gas particles."""

import numpy as np
import pytest

from jet_ism import BOLTZMANN, GAMMA, PROTONMASS, mu, unit_velocity
from jet_ism.gas.general import get_temp

X_H = 0.76  # hydrogen mass fraction, hardcoded in get_temp


# ---------------------------------------------------------------------------
# Helpers that re-implement the formula independently so tests don't just
# copy the implementation — they express the physics directly.
# ---------------------------------------------------------------------------

def _expected_uniform(u):
    """T [K] for uniform mean molecular weight."""
    return (GAMMA - 1) * u * PROTONMASS * mu / BOLTZMANN * unit_velocity**2


def _expected_local(u, n_e):
    """T [K] using per-particle electron abundance to compute mu_local."""
    mu_local = 4 / (3 * X_H + 1 + 4 * X_H * n_e)
    return (GAMMA - 1) * u * PROTONMASS * mu_local / BOLTZMANN * unit_velocity**2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetTempUniform:
    def test_known_value_single_particle(self, make_snap):
        u = 200.0  # (km/s)^2 — typical ISM internal energy
        snap = make_snap(internal_energy=u, electron_abundance=0.0)
        T = get_temp(snap, approach="uniform")
        assert T == pytest.approx(_expected_uniform(u), rel=1e-10)

    def test_temperature_scales_linearly_with_internal_energy(self, make_snap):
        snap1 = make_snap(internal_energy=100.0, electron_abundance=0.0)
        snap2 = make_snap(internal_energy=300.0, electron_abundance=0.0)
        T1 = get_temp(snap1, approach="uniform")
        T2 = get_temp(snap2, approach="uniform")
        assert T2 == pytest.approx(3 * T1, rel=1e-10)

    def test_output_shape_matches_input(self, make_snap):
        u = np.array([100.0, 200.0, 500.0])
        snap = make_snap(internal_energy=u, electron_abundance=np.zeros(3))
        T = get_temp(snap, approach="uniform")
        assert T.shape == (3,)

    def test_uniform_ignores_electron_abundance(self, make_snap):
        u = 200.0
        snap_zero = make_snap(internal_energy=u, electron_abundance=0.0)
        snap_one  = make_snap(internal_energy=u, electron_abundance=1.0)
        T_zero = get_temp(snap_zero, approach="uniform")
        T_one  = get_temp(snap_one,  approach="uniform")
        assert T_zero == pytest.approx(T_one, rel=1e-12)


class TestGetTempLocal:
    def test_neutral_gas_known_value(self, make_snap):
        """Fully neutral gas (n_e=0): mu_local = 4/(3X+1) > mu_uniform."""
        u, n_e = 200.0, 0.0
        snap = make_snap(internal_energy=u, electron_abundance=n_e)
        T = get_temp(snap, approach="local")
        assert T == pytest.approx(_expected_local(u, n_e), rel=1e-10)

    def test_fully_ionized_known_value(self, make_snap):
        """Fully ionized (n_e=1): mu_local = 4/(3X+1+4X)."""
        u, n_e = 200.0, 1.0
        snap = make_snap(internal_energy=u, electron_abundance=n_e)
        T = get_temp(snap, approach="local")
        assert T == pytest.approx(_expected_local(u, n_e), rel=1e-10)

    def test_neutral_gas_hotter_than_uniform(self, make_snap):
        """Neutral gas has higher mu_local than mu_uniform, so T_local > T_uniform."""
        u = 200.0
        snap = make_snap(internal_energy=u, electron_abundance=0.0)
        T_uniform = get_temp(snap, approach="uniform")
        T_local   = get_temp(snap, approach="local")
        assert T_local > T_uniform

    def test_temperature_increases_with_lower_ionization(self, make_snap):
        """Lower electron abundance → higher mu_local → higher temperature."""
        u = 200.0
        snap_neutral  = make_snap(internal_energy=u, electron_abundance=0.0)
        snap_ionized  = make_snap(internal_energy=u, electron_abundance=1.0)
        T_neutral = get_temp(snap_neutral, approach="local")
        T_ionized = get_temp(snap_ionized, approach="local")
        assert T_neutral > T_ionized

    def test_array_of_mixed_ionization(self, make_snap):
        u   = np.array([200.0, 200.0, 200.0])
        n_e = np.array([0.0,   0.5,   1.0])
        snap = make_snap(internal_energy=u, electron_abundance=n_e)
        T = get_temp(snap, approach="local")
        assert T.shape == (3,)
        expected = _expected_local(u, n_e)
        np.testing.assert_allclose(T, expected, rtol=1e-10)


class TestGetTempPhysical:
    def test_temperatures_are_positive(self, make_snap):
        u   = np.array([50.0, 200.0, 1000.0])
        n_e = np.array([0.0,  0.5,   1.0])
        snap = make_snap(internal_energy=u, electron_abundance=n_e)
        for approach in ("uniform", "local"):
            T = get_temp(snap, approach=approach)
            assert np.all(T > 0), f"Non-positive temperature with approach='{approach}'"

    def test_reasonable_temperature_range(self, make_snap):
        """u=200 (km/s)^2 should give ~10^4 K, well within ISM range."""
        snap = make_snap(internal_energy=200.0, electron_abundance=1.0)
        T = get_temp(snap, approach="local")
        assert 1e3 < T.item() < 1e5

    def test_invalid_approach_raises(self, make_snap):
        snap = make_snap(internal_energy=200.0, electron_abundance=1.0)
        with pytest.raises(ValueError):
            get_temp(snap, approach="bad_approach")
