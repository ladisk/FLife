"""Criteria of the form trace(Q S): EVMS, thermoelastic and Lemaitre (synthetic PSDs only).
The critical-plane criteria a^T S a are included in the real-PSD checks.

Preumont A, Piefort V. Predicting random high-cycle fatigue life with finite elements.
J Vib Acoust 1994;116:245-248.

Sonc J, Zaletelj K, Slavic J. Application of thermoelasticity in the frequency-domain
multiaxial vibration-fatigue criterion. Mech Syst Signal Process 2025.

Ge J, Sun Y, Zhou S. Fatigue life estimation under multiaxial random loading by means of the
equivalent Lemaitre stress and multiaxial S-N curve methods. Int J Fatigue 2015;79:65-74.
"""
import warnings
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.multiaxial import criteria

F = np.arange(0, 201, 1.)
P = np.exp(-0.5 * ((F - 50) / 8)**2)
P[0] = 0.

CRITERIA = [('EVMS', {}), ('thermoelastic', {}), ('Lemaitre', {'poisson_ratio': 0.3})]
PLANE_CRITERIA = [('max_normal', {}), ('max_shear', {}), ('max_normal_and_shear', {'s_af': 300., 'tau_af': 180.}),
                  ('gao_mwcm', {'s_af': 300., 'tau_af': 180.})]


def random_psd(n_comp, n_points=None, seed=0):
    """Hermitian PSD matrix of two coherent sources with complex mode shapes, (f, n, n) or (N, f, n, n)."""
    rng = np.random.default_rng(seed)
    shape = (2, n_comp) if n_points is None else (n_points, 2, n_comp)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    P2 = np.roll(P, 30)
    return (np.einsum('...i,...j,f->...fij', a[..., 0, :], a[..., 0, :].conj(), P)
            + np.einsum('...i,...j,f->...fij', a[..., 1, :], a[..., 1, :].conj(), P2))


@pytest.mark.parametrize('name, kwargs', CRITERIA + PLANE_CRITERIA)
@pytest.mark.parametrize('n_comp', [3, 6])
def test_single_point_psd_is_real(name, kwargs, n_comp):
    es = FLife.EquivalentStress(input={'PSD': random_psd(n_comp), 'f': F})
    getattr(es, name)(**kwargs)
    assert np.isrealobj(es.psd)
    assert np.all(es.psd[:, 1] >= -1e-12 * es.psd[:, 1].max())


@pytest.mark.parametrize('name, kwargs', CRITERIA + PLANE_CRITERIA)
@pytest.mark.parametrize('n_comp', [3, 6])
def test_multipoint_psd_is_real_without_warning(name, kwargs, n_comp):
    es = FLife.EquivalentStress(input={'PSD': random_psd(n_comp, n_points=3), 'f': F})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        getattr(es, name)(**kwargs)
    assert np.isrealobj(es.eq_psd_multipoint[0])


@pytest.mark.parametrize('name, kwargs', CRITERIA)
@pytest.mark.parametrize('n_comp', [3, 6])
def test_criterion_accepts_all_points_at_once(name, kwargs, n_comp):
    S = random_psd(n_comp, n_points=4, seed=3)
    criterion = getattr(criteria, '_' + name)
    all_points = criterion(None, S, **kwargs)
    assert all_points.shape == S.shape[:2]
    for i in range(len(S)):
        np.testing.assert_allclose(all_points[i], criterion(None, S[i], **kwargs), rtol=1e-12)


# per-point sum over frequency and one value, computed with FLife 2.x (loop over points)
REFERENCE = {
    'EVMS': ([5765.911460753014, 5445.473962386703, 5034.3943973267305, 4547.138532397865], 15.500192425609512),
    'thermoelastic': ([2869.6633516325246, 2536.8118668064376, 2191.561316386067, 1844.7939245507425], 4.843617885080692),
    'Lemaitre': ([5379.745046203616, 5057.652349642668, 4655.349986534643, 4186.825918018249], 14.07931582020567),
}


@pytest.mark.parametrize('name, kwargs', CRITERIA)
@pytest.mark.parametrize('dim', ['2D', '3D'])
def test_reference_values(name, kwargs, dim):
    S = np.load(os.path.join(my_path, '..', 'data', f'test_multiaxial_PSD_{dim}.npy'))[331:335]
    freq = np.arange(0, 240, 3)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': freq})
    getattr(es, name)(**kwargs)
    sums, value = REFERENCE[name]
    np.testing.assert_allclose(es.eq_psd_multipoint[0].sum(axis=1), sums, rtol=1e-12)
    np.testing.assert_allclose(es.eq_psd_multipoint[0][2, 30], value, rtol=1e-12)
