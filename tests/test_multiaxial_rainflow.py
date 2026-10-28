"""Multiaxial rainflow criterion against analytic references (synthetic PSDs only).

Pitoiset X, Preumont A. Spectral methods for multiaxial random fatigue analysis of
metallic structures. Int J Fatigue 2000;22:541-550.
"""
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


def _search(S):
    """Constrained numerical search for max of c^T Re(S) c on |c| = 1, at each frequency."""
    from scipy.optimize import minimize
    cons = {'type': 'eq', 'fun': lambda c: np.sum(c**2) - 1}
    out = np.zeros(len(S))
    for i, s in enumerate(np.real(S)):
        scale = np.trace(s)
        if scale > 0:
            res = minimize(lambda c: -(c @ s @ c) / scale, np.ones(3) / np.sqrt(3),
                           constraints=[cons], method='SLSQP')
            c = res.x / np.linalg.norm(res.x)
            out[i] = c @ s @ c
    return out


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_multiaxial_rainflow_known_principal_values(seed):
    # Re(S) = R diag(l1, l2, l3) R^T: the maximum of c^T Re(S) c on |c| = 1 is max(l1, l2, l3).
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    lam = np.stack([P1, 0.7 * P2, 0.4 * P1 + 0.3 * P2], axis=1)
    S = np.einsum('ik,fk,jk->fij', R, lam, R)
    np.testing.assert_allclose(eq_psd(S), lam.max(axis=1), rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_multiaxial_rainflow_is_max_over_directions(seed):
    # At each frequency the result is the maximum of c^T Re(S) c over unit vectors c.
    S = random_state(seed)
    eq = eq_psd(S)
    scan = np.einsum('ni,fij,nj->fn', _scan_directions(), np.real(S), _scan_directions()).max(axis=1)
    assert np.all(eq >= scan * (1 - 1e-12))
    np.testing.assert_allclose(eq, scan, rtol=1e-3, atol=1e-15)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_multiaxial_rainflow_equals_numerical_search(seed):
    S = random_state(seed)
    np.testing.assert_allclose(eq_psd(S), _search(S), rtol=1e-6, atol=1e-15)


def test_multiaxial_rainflow_multipoint_equals_single_point():
    S = np.stack([random_state(s) for s in range(3)])
    np.testing.assert_allclose(eq_psd(S), np.stack([eq_psd(s) for s in S]), rtol=1e-12)


def test_multiaxial_rainflow_rejects_3d_stress():
    with pytest.raises(ValueError, match='biaxial'):
        eq_psd(np.einsum('ij,f->fij', np.eye(6), P1))
