"""Carpinteri-Spagnoli criterion against analytic references (synthetic PSDs only).

Carpinteri A, Spagnoli A, Vantadori S. Reformulation in the frequency domain of a
critical plane-based multiaxial fatigue criterion. Int J Fatigue 2014;67:55-61.
"""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.multiaxial import cplane

F = np.arange(0, 201, 1.)
P = np.exp(-0.5 * ((F - 50) / 8)**2) + 0.5 * np.exp(-0.5 * ((F - 120) / 6)**2)
P[0] = 0.
P1 = np.exp(-0.5 * ((F - 50) / 8)**2)
P2 = np.exp(-0.5 * ((F - 120) / 6)**2)
P1[0] = P2[0] = 0.

S_AF, TAU_AF = 300., 180.          # 1/sqrt(3) < tau_af/s_af < 1, so 0 < delta < pi/4
DELTA = 3 * np.pi / 8 * (1 - (TAU_AF / S_AF)**2)

UNIAXIAL = np.diag([1., 0, 0])
SHEAR = np.array([[0., 1, 0], [1, 0, 0], [0, 0, 0]])


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


def psd_from_sources(tensors, psds, n=len(F)):
    """PSD matrix (f,6,6) of a sum of fully coherent sources: complex stress tensor times scalar PSD."""
    S = np.zeros((n, 6, 6), dtype=complex)
    for T, p in zip(tensors, psds):
        v = voigt(T)
        S += p[:, None, None] * np.outer(v, np.conj(v))[None]
    return S


def random_state(seed):
    rng = np.random.default_rng(seed)
    tensors = []
    for _ in range(2):
        B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        tensors.append((B + B.T) / 2)
    return tensors


def cs_psd(S, f=F, T=1., **kwargs):
    """Equivalent C-S PSD; T is given to EquivalentStress (the only way on older versions)."""
    es = FLife.EquivalentStress(input={"PSD": S, "f": f}, T=T, fs=10 * f.max())
    es.cs(**kwargs)
    return es.psd[:, 1]


def ratio(s_eq):
    """Equivalent PSD over the input PSD P (where P is not negligible)."""
    m = P > 1e-6 * P.max()
    return np.real(s_eq[m]) / P[m]


def uniaxial_factor():
    """Uniaxial: normal and shear stress on the plane tilted by delta (Eq. 19), Eq. (22) as printed."""
    return np.cos(DELTA)**4 + S_AF / TAU_AF * np.sin(DELTA)**2 * np.cos(DELTA)**2


def test_cs_frequency_vector():
    """The spectral moments, Eq. (15a/b), use the given frequency vector."""
    # np.arange(0, N*df, df) gives N+1 points here (202*0.1 = 20.200000000000003)
    f = np.arange(202) * 0.1
    S = psd_from_sources([UNIAXIAL], [np.ones(len(f))], n=len(f))
    np.testing.assert_allclose(cs_psd(S, f=f, T=10., s_af=1., tau_af=1.), 1.)
    assert cplane.csrandom(S, 0.1, 1., 1.).shape == (6, 6)


def test_cs_global_plane_search():
    """1-hat is the global maximum of Eq. (17) over 0 <= phi <= 2 pi, 0 <= theta <= pi (Sec. 3).

    Uncorrelated sigma_x (variance 1) and sigma_z (variance 0.9) at the same
    frequency: both axes are local maxima, sigma_x is the global one.
    """
    S = np.zeros((len(F), 6, 6))
    S[:, 0, 0], S[:, 2, 2] = P, 0.9 * P
    np.testing.assert_allclose(cs_psd(S, s_af=1., tau_af=1.), P, atol=1e-10)


@pytest.mark.parametrize('R', ROTATIONS[:4])
def test_cs_uniaxial(R):
    S = psd_from_sources([R @ UNIAXIAL @ R.T], [P])
    np.testing.assert_allclose(ratio(cs_psd(S, s_af=S_AF, tau_af=TAU_AF)), uniaxial_factor(), rtol=1e-6)   # plane search to ~1e-8 rad


def test_cs_uniaxial_near_phi_zero():
    """1-hat just below phi = 0: phi is periodic, the local refinement must cross phi = 0."""
    phi, gamma = -np.radians(2.), np.radians(60.)
    n = np.array([np.sin(gamma) * np.cos(phi), np.sin(gamma) * np.sin(phi), np.cos(gamma)])
    S = psd_from_sources([np.outer(n, n)], [P])
    np.testing.assert_allclose(ratio(cs_psd(S, s_af=S_AF, tau_af=TAU_AF)), uniaxial_factor(), rtol=1e-6)


def test_cs_scale_invariance():
    S = psd_from_sources(random_state(3), [P1, P2])
    ref = cs_psd(S, s_af=S_AF, tau_af=TAU_AF)
    for c in [1e-8, 1e8]:
        np.testing.assert_allclose(cs_psd(c * S, s_af=S_AF, tau_af=TAU_AF), c * ref, rtol=1e-7, atol=1e-9 * c * ref.max())


@pytest.mark.xfail(strict=True, reason='C-S: the sign of 3-hat (tilt direction) and 3-hat itself under '
                   'proportional loading are not defined by the source; the frame convention of '
                   'C_S_Spettrale.m is kept, so results depend on the reference frame. Maintainer decision.')
@pytest.mark.parametrize('case', ['general', 'pure shear'])
def test_cs_frame_independence(case):
    if case == 'general':
        tensors, psds = random_state(1), [P1, P2]
    else:
        tensors, psds = [SHEAR], [P]
    ref = cs_psd(psd_from_sources(tensors, psds), s_af=S_AF, tau_af=TAU_AF)
    for R in ROTATIONS[:4]:
        S = psd_from_sources([R @ T @ R.T for T in tensors], psds)
        np.testing.assert_allclose(cs_psd(S, s_af=S_AF, tau_af=TAU_AF), ref, rtol=1e-6)
