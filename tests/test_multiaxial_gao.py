"""Spectral MWCM of Gao et al. (2021) against closed-form references (synthetic PSDs only).

Gao D, Yao W, Wen W, Huang J. A multiaxial fatigue life prediction method for metallic material
under combined random vibration loading and mean stress loading in the frequency domain.
Int J Fatigue 2021;148:106235.

Susmel L, Lazzarin P. A bi-parametric Wohler curve for high cycle multiaxial fatigue assessment.
Fatigue Fract Eng Mater Struct 2002;25:63-78.
"""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife


F = np.arange(1., 101.)
DF = F[1] - F[0]
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
TAU_AF = 100.
S_AF = np.sqrt(3) * TAU_AF


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


def rotate(S, rng):
    T = voigt_T(random_rotation(rng))
    return np.einsum('ij,...fjk,lk->...fil', T, S, T)


def single_component(i, n=6, variance=1.):
    """Gaussian band PSD in Voigt component i only, scaled to the given variance."""
    S = np.zeros((len(F), n, n))
    P = np.exp(-0.5 * ((F - 50) / 5)**2)
    S[:, i, i] = variance * P / (np.sum(P) * DF)
    return S


def random_psd(rng, n_points=None):
    """Hermitian PSD of six coherent sources with complex mode shapes, (f, 6, 6) or (N, f, 6, 6)."""
    shape = (6,) if n_points is None else (n_points, 6)
    S = np.zeros(shape[:-1] + (len(F), 6, 6), dtype=complex)
    for _ in range(6):
        v = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        P = np.exp(-0.5 * ((F - rng.uniform(10, 90)) / rng.uniform(3, 10))**2)
        S += np.einsum('...i,...j,f->...fij', v, v.conj(), P)
    return S


def eq_psd(S, s_af=S_AF, tau_af=TAU_AF):
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.gao_mwcm(s_af=s_af, tau_af=tau_af)
    return es


def m0(S, s_af=S_AF):
    """Variance of the equivalent stress, through the multipoint path."""
    return np.sum(np.real(eq_psd(S[None], s_af=s_af).eq_psd_multipoint[0][0])) * DF


@pytest.mark.parametrize('n, i', [(6, 3), (6, 4), (6, 5), (3, 2)])
def test_pure_torsion(n, i):
    # sigma_c = 0 on the maximum shear plane, so G_eq = G_tau
    S = single_component(i, n)
    np.testing.assert_allclose(eq_psd(S).psd[:, 1], S[:, i, i], rtol=1e-6, atol=1e-12 * S.max())


@pytest.mark.parametrize('r', [np.sqrt(3), 1.5, 2.])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_uniaxial_in_any_direction(r, seed):
    # tau_c = sigma_c = sigma/2 on the maximum shear plane, q = 1 - r/2 (Susmel and Lazzarin 2002, Eq. 11),
    # so G_eq = G_sigma (1 + q)^2 / 4 = G_sigma (2 - r/2)^2 / 4
    S = single_component(0)
    S_rot = rotate(S, np.random.default_rng(seed))
    expected = np.sum(S[:, 0, 0]) * DF * (2 - r / 2)**2 / 4
    np.testing.assert_allclose(m0(S_rot, s_af=r * TAU_AF), expected, rtol=1e-5)


def test_torsion_fatigue_limit():
    # Susmel and Lazzarin (2002), Eq. 11, pure torsion: tau_a = tau_-1 is the limit; the linear form is exact
    S = single_component(3, variance=TAU_AF**2)
    np.testing.assert_allclose(np.sqrt(m0(S)), TAU_AF, rtol=1e-6)


def test_uniaxial_fatigue_limit_within_linearisation():
    # Susmel and Lazzarin (2002), Eq. 11, fully reversed uniaxial sigma_a = s_af: tau_a = sigma_n,max = s_af/2,
    # tau_a + (tau_af - s_af/2) * 1 = tau_af exactly. The linear form of Gao (2021), Eq. 16, gives
    # (1 + q) s_af / 2 = r (2 - r/2) / 2 * tau_af, 1.8 % below tau_af for r = sqrt(3). Tolerance 2 %.
    s_a = S_AF
    tau_a = sigma_n_max = s_a / 2
    susmel = tau_a + (TAU_AF - S_AF / 2) * sigma_n_max / tau_a
    np.testing.assert_allclose(susmel, TAU_AF, rtol=1e-12)

    S = single_component(0, variance=s_a**2)
    tau_eq = np.sqrt(m0(S))
    np.testing.assert_allclose(tau_eq, np.sqrt(3) * (2 - np.sqrt(3) / 2) / 2 * TAU_AF, rtol=1e-5)
    assert abs(tau_eq / susmel - 1) < 0.02
    assert abs(tau_eq / susmel - 1) > 0.017


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_rotation_invariance(seed):
    rng = np.random.default_rng(seed)
    S = random_psd(rng)
    np.testing.assert_allclose(m0(rotate(S, rng)), m0(S), rtol=1e-4)


def test_single_point_equals_multipoint():
    S = random_psd(np.random.default_rng(3), n_points=3)
    multi = eq_psd(S).eq_psd_multipoint[0]
    for i in range(len(S)):
        np.testing.assert_allclose(multi[i], eq_psd(S[i]).psd[:, 1], rtol=1e-12)


@pytest.mark.parametrize('n_comp', [3, 6])
def test_output_is_real_and_non_negative(n_comp):
    S = random_psd(np.random.default_rng(4), n_points=2)
    if n_comp == 3:
        S = S[..., [0, 1, 3], :][..., [0, 1, 3]]
    multi = eq_psd(S).eq_psd_multipoint[0]
    single = eq_psd(S[0]).psd[:, 1]
    for G in (multi, single):
        assert np.isrealobj(G)
        assert np.all(G >= -1e-12 * G.max())
