"""Multiaxial rainflow criterion against analytic references (synthetic PSDs only).

Pitoiset X, Preumont A. Spectral methods for multiaxial random fatigue analysis of
metallic structures. Int J Fatigue 2000;22:541-550.

Lutes LD, Larsen CE. Improved spectral method for variable amplitude fatigue
prediction. J Struct Eng 1990;116:1149-1164.
"""
import warnings
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

F = np.arange(0, 201, 1.)
P1 = np.exp(-0.5 * ((F - 50) / 8)**2)
P2 = np.exp(-0.5 * ((F - 120) / 6)**2)
P1[0] = P2[0] = 0.


def random_state(seed):
    """Plane-stress PSD matrix (f, 3, 3) of two coherent sources in separate bands."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    return (np.einsum('i,j,f->fij', a[0], a[0].conj(), P1)
            + np.einsum('i,j,f->fij', a[1], a[1].conj(), P2))


def eq_psd(S, f=F, **kwargs):
    kwargs.setdefault('k', 5.)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': f})
    es.multiaxial_rainflow(**kwargs)
    return es.eq_psd_multipoint[0] if es.multipoint else es.psd[:, 1]


@pytest.mark.parametrize('scale', [1e-8, 1e8, 1e12])
@pytest.mark.parametrize('seed', [0, 1])
def test_multiaxial_rainflow_scales_with_the_input(scale, seed):
    # A change of stress unit (e.g. MPa to Pa, scale 1e12 on the PSD) only scales the result.
    S = random_state(seed)
    np.testing.assert_allclose(eq_psd(scale * S), scale * eq_psd(S), rtol=1e-5)


def _scan_directions(n=40000):
    """Unit vectors spread over the sphere."""
    c = np.random.default_rng(1).normal(size=(n, 3))
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def test_multiaxial_rainflow_multipoint_equals_single_point():
    S = np.stack([random_state(s) for s in range(3)])
    np.testing.assert_allclose(eq_psd(S), np.stack([eq_psd(s) for s in S]), rtol=1e-12)


def test_multiaxial_rainflow_rejects_3d_stress():
    with pytest.raises(ValueError, match='biaxial'):
        eq_psd(np.einsum('ij,f->fij', np.eye(6), P1))


def _moment(S, c, q):
    """2/k-th spectral moment lambda_q(c) of c^T Re(S) c, for each row of c."""
    psd = np.einsum('ni,fij,nj->nf', c, np.real(S), c)
    return np.trapezoid(psd * F**q, F, axis=1) if hasattr(np, 'trapezoid') else np.trapz(psd * F**q, F, axis=1)


@pytest.mark.parametrize('i', [0, 1, 2])
@pytest.mark.parametrize('k', [None, 3., 7.3])
def test_multiaxial_rainflow_uniaxial_reduction(i, k):
    S = np.zeros((len(F), 3, 3))
    S[:, i, i] = P1 + P2
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        es.multiaxial_rainflow(k=k)
    np.testing.assert_allclose(es.psd[:, 1], P1 + P2, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(es.multiaxial_rainflow_direction, np.eye(3)[i], atol=1e-12)


@pytest.mark.parametrize('k', [3., 7.3])
def test_multiaxial_rainflow_proportional_loading(k):
    # s(t) = a x(t): c = a / |a| and the PSD of c^T s is |a|^2 P (Eq. 14)
    a = np.array([2., -1., 0.5])
    S = np.einsum('i,j,f->fij', a, a, P1 + P2)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.multiaxial_rainflow(k=k)
    np.testing.assert_allclose(es.multiaxial_rainflow_direction, a / np.linalg.norm(a), atol=1e-12)
    np.testing.assert_allclose(es.psd[:, 1], a @ a * (P1 + P2), rtol=1e-12, atol=1e-15)


def test_multiaxial_rainflow_two_bands_fixed_direction():
    # s_x at 50 Hz and uncorrelated shear at 120 Hz: one constant c (Eq. 9) cannot take both bands
    S = np.zeros((len(F), 3, 3))
    S[:, 0, 0] = P1
    S[:, 2, 2] = 0.8 * P2
    with pytest.warns(UserWarning):
        # maximum variance: the larger, low-frequency band s_x
        np.testing.assert_allclose(eq_psd(S, k=None), P1, rtol=1e-12, atol=1e-15)
    # maximum single moment damage at k = 3: the high-frequency shear band
    np.testing.assert_allclose(eq_psd(S, k=3.), 0.8 * P2, rtol=1e-12, atol=1e-15)
    envelope = np.linalg.eigvalsh(S)[:, -1]
    assert eq_psd(S, k=3.).sum() < 0.9 * envelope.sum()


@pytest.mark.parametrize('k', [3., 5., 10.])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_multiaxial_rainflow_is_max_single_moment_over_directions(k, seed):
    # Sec. 3 and 4.1: c is the unit vector with the largest single moment damage
    S = random_state(seed)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.multiaxial_rainflow(k=k)
    c = es.multiaxial_rainflow_direction
    np.testing.assert_allclose(np.linalg.norm(c), 1., rtol=1e-12)
    np.testing.assert_allclose(es.psd[:, 1], np.einsum('i,fij,j->f', c, np.real(S), c), rtol=1e-12)
    q = 2. / k
    best = _moment(S, c[None], q)[0]
    scan = _moment(S, _scan_directions(), q)
    assert best >= scan.max() * (1 - 1e-12)
    assert best <= scan.max() * (1 + 1e-3)


@pytest.mark.parametrize('seed', [0, 1])
def test_multiaxial_rainflow_has_shortest_single_moment_life(seed):
    # FLife.SingleMoment of the result against random unit vectors c
    C, k = 1.8e22, 7.3
    S = random_state(seed)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.multiaxial_rainflow(k=k)
    life = FLife.SingleMoment(es).get_life(C, k)
    for c in _scan_directions(50):
        sd = FLife.SpectralData(input={'PSD': np.einsum('i,fij,j->f', c, np.real(S), c), 'f': F})
        assert life <= FLife.SingleMoment(sd).get_life(C, k) * (1 + 1e-10)


def test_multiaxial_rainflow_without_k_warns():
    with pytest.warns(UserWarning, match='`k`'):
        eq_psd(random_state(0), k=None)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        eq_psd(random_state(0), k=4.)


def test_multiaxial_rainflow_rejects_non_positive_k():
    with pytest.raises(ValueError, match='positive'):
        eq_psd(random_state(0), k=0.)


def test_multiaxial_rainflow_only_real_part_and_sign_of_c():
    S = random_state(3)
    A = np.random.default_rng(4).normal(size=(3, 3))
    S_im = S + 1j * np.einsum('ij,f->fij', A - A.T, P1)
    np.testing.assert_allclose(eq_psd(S_im), eq_psd(S), rtol=1e-12)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.multiaxial_rainflow(k=4.)
    c = es.multiaxial_rainflow_direction
    np.testing.assert_allclose(np.einsum('i,fij,j->f', -c, np.real(S), -c), es.psd[:, 1], rtol=1e-12)


def test_multiaxial_rainflow_pure_shear_depends_on_frame():
    # The method is not frame invariant: pure shear tau gives tau^2 P in the shear frame
    # (c = e3) and 2 tau^2 P at 45 deg (s_x = tau, s_y = -tau, c = (1, -1, 0)/sqrt(2)).
    P = P1 + P2
    shear = np.zeros((len(F), 3, 3))
    shear[:, 2, 2] = P
    np.testing.assert_allclose(eq_psd(shear), P, rtol=1e-12, atol=1e-15)
    a = np.array([1., -1., 0.])
    np.testing.assert_allclose(eq_psd(np.einsum('i,j,f->fij', a, a, P)), 2 * P, rtol=1e-12, atol=1e-15)


def test_multiaxial_rainflow_multipoint_direction():
    S = np.stack([random_state(s) for s in range(3)])
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.multiaxial_rainflow(k=5.)
    for s, c in zip(S, es.multiaxial_rainflow_direction):
        one = FLife.EquivalentStress(input={'PSD': s, 'f': F})
        one.multiaxial_rainflow(k=5.)
        np.testing.assert_allclose(c, one.multiaxial_rainflow_direction, atol=1e-12)
