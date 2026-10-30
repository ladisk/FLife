"""Nieslony criterion against analytic references (synthetic PSDs only).

Niesłony A, Böhm M, Owsiński R, Dziura A, Czekaj K. Integrating von Mises and hydrostatic
stresses in frequency domain multiaxial fatigue criteria for vibration fatigue analysis.
Mech Syst Signal Process 2025;224:112229.
"""
import warnings
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

F = np.arange(0, 201, 1.)
P = np.exp(-0.5 * ((F - 50) / 8)**2)
P[0] = 0.

S_AF, TAU_AF = 300., 180.

UNIAXIAL = np.diag([1., 0, 0])
SHEAR = np.array([[0., 1, 0], [1, 0, 0], [0, 0, 0]])
HYDROSTATIC = np.eye(3)


def rotation(a, b, c):
    """Rotation matrix from ZYZ angles."""
    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    def ry(t):
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    return rz(a) @ ry(b) @ rz(c)


ROTATIONS = [rotation(*a) for a in np.random.default_rng(0).uniform(0, 2*np.pi, size=(10, 3))]


def voigt(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]])


def nieslony(T, dtype=float, coefficient_load_type='tension'):
    """Equivalent PSD of a proportional stress state T with PSD P."""
    v = voigt(T)
    S = (P[:, None, None] * np.outer(v, v)[None]).astype(dtype)
    eq = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    eq.Nieslony(S_AF, TAU_AF, coefficient_load_type=coefficient_load_type)
    return eq.psd[:, 1]


@pytest.mark.parametrize('dtype', [float, complex])
def test_rotated_shear_is_finite_and_invariant(dtype):
    # The hydrostatic PSD of pure shear is 0; after a rotation it is round-off, which may be
    # negative. Eqs. (27)-(31): G_p is an auto-PSD, so sqrt(G_p*G_EMS) must not give NaN.
    ref = nieslony(SHEAR, dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        for R in ROTATIONS:
            s_eq = nieslony(R @ SHEAR @ R.T, dtype)
            assert np.all(np.isfinite(s_eq))
            np.testing.assert_allclose(s_eq, ref, rtol=1e-8, atol=1e-8 * ref.max())


def test_shear():
    # Eq. (33) and (38): pure shear gives G_eq = 3 B^2 G_tau
    np.testing.assert_allclose(nieslony(SHEAR), (S_AF / TAU_AF)**2 * P, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(nieslony(SHEAR, coefficient_load_type='torsion'), P, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('R', [np.eye(3)] + ROTATIONS[:3])
def test_uniaxial(R):
    # Eq. (32): uniaxial tension with the tension coefficients gives G_eq = G_xx
    np.testing.assert_allclose(nieslony(R @ UNIAXIAL @ R.T), P, rtol=1e-8, atol=1e-8)


def test_hydrostatic():
    # G_EMS = 0 and r_p,EMS = 1, so Eq. (21) with Eq. (35) gives G_eq = A^2 G_p, G_p = G_xx
    A = 3 - np.sqrt(3) * S_AF / TAU_AF
    np.testing.assert_allclose(nieslony(HYDROSTATIC), A**2 * P, rtol=1e-10, atol=1e-12)
