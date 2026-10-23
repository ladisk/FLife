import numpy as np
import pytest

import FLife


F = np.arange(1., 101.)
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
S_AF, TAU_AF = 1., 0.6


def voigt_T(R):
    """6x6 matrix T with s' = T s for sigma' = R sigma R^T."""
    T = np.zeros((6, 6))
    for j, (a, b) in enumerate(VOIGT):
        E = np.zeros((3, 3))
        E[a, b] = E[b, a] = 1
        Ep = R @ E @ R.T
        T[:, j] = [Ep[i, k] for (i, k) in VOIGT]
    return T


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def m0_eq(S):
    es = FLife.EquivalentStress(input={'PSD': S[None], 'f': F}, T=1, fs=5000)
    es.max_normal_and_shear(s_af=S_AF, tau_af=TAU_AF)
    return np.sum(np.real(es.eq_psd_multipoint[0][0]))


def single_component(i):
    S = np.zeros((len(F), 6, 6))
    S[:, i, i] = np.exp(-0.5 * ((F - 50) / 5)**2)
    return S


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_uniaxial_stress_in_any_direction(seed):
    # Nieslony and Macha (2007), Eq. (3.23): with K = 2 tau_af / s_af - 1 the equivalent stress of
    # uniaxial loading is the uniaxial stress, in whatever direction it acts
    S = single_component(0)
    T = voigt_T(random_rotation(np.random.default_rng(seed)))
    S_rot = np.einsum('ij,fjk,lk->fil', T, S, T)
    np.testing.assert_allclose(m0_eq(S_rot), np.sum(S[:, 0, 0]), rtol=1e-5)


def test_pure_shear():
    # torsion: sigma_eq = tau * s_af / tau_af on the maximum shear plane
    S = single_component(3)
    np.testing.assert_allclose(m0_eq(S), np.sum(S[:, 3, 3]) * (S_AF / TAU_AF)**2, rtol=1e-5)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_rotation_invariance(seed):
    rng = np.random.default_rng(seed)
    S = np.zeros((len(F), 6, 6), dtype=complex)
    for _ in range(6):
        v = rng.normal(size=6) + 1j * rng.normal(size=6)
        S += np.exp(-0.5 * ((F - rng.uniform(10, 90)) / rng.uniform(3, 10))**2)[:, None, None] * np.outer(v, v.conj())
    T = voigt_T(random_rotation(rng))
    np.testing.assert_allclose(m0_eq(np.einsum('ij,fjk,lk->fil', T, S, T)), m0_eq(S), rtol=1e-4)
