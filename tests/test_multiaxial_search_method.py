"""The `search_method` of the critical-plane criteria reaches the plane search for single and multipoint input."""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.multiaxial import cplane

F = np.arange(0, 201, 1.)
P = np.exp(-0.5 * ((F - 50) / 8)**2)
P[0] = 0.

CRITERIA = [('max_normal', {}), ('max_shear', {}), ('max_normal_and_shear', {'s_af': 300., 'tau_af': 180.})]


def psd(n_points=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (3,) if n_points is None else (n_points, 3)
    a = rng.normal(size=shape)
    return np.einsum('...i,...j,f->...fij', a, a, P)


@pytest.mark.parametrize('n_points', [None, 2])
@pytest.mark.parametrize('name, kwargs', CRITERIA)
@pytest.mark.parametrize('search_method', ['local', 'global'])
def test_search_method_is_passed_to_plane_search(monkeypatch, n_points, name, kwargs, search_method):
    used = []
    max_variance = cplane.max_variance

    def spy(*args, **kw):
        used.append(kw.get('search_method'))
        return max_variance(*args, **dict(kw, search_method='local'))

    monkeypatch.setattr(cplane, 'max_variance', spy)
    es = FLife.EquivalentStress(input={'PSD': psd(n_points), 'f': F})
    getattr(es, name)(search_method=search_method, **kwargs)
    assert used == [search_method] * (n_points or 1)


def test_multipoint_global_equals_single_point_global():
    S = psd(n_points=2, seed=1)
    multi = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    multi.max_normal(search_method='global')
    for i in range(2):
        single = FLife.EquivalentStress(input={'PSD': S[i], 'f': F})
        single.max_normal(search_method='global')
        np.testing.assert_allclose(multi.eq_psd_multipoint[0][i], single.psd[:, 1], rtol=1e-12)
