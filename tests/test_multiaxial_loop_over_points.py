"""`EquivalentStress.loop_over_points` is deprecated: it warns and gives the same result as before,
and no criterion method calls it."""
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


def random_psd(n_comp, n_points=None, seed=0):
    """Hermitian PSD matrix of two coherent sources with complex mode shapes, (f, n, n) or (N, f, n, n)."""
    rng = np.random.default_rng(seed)
    shape = (2, n_comp) if n_points is None else (n_points, 2, n_comp)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    P2 = np.roll(P, 30)
    return (np.einsum('...i,...j,f->...fij', a[..., 0, :], a[..., 0, :].conj(), P)
            + np.einsum('...i,...j,f->...fij', a[..., 1, :], a[..., 1, :].conj(), P2))


def old_loop_over_points(es, criterion, **kwargs):
    """`loop_over_points` as it was before the deprecation."""
    s, f = es.multiaxial_psd
    return np.array([criterion(es, s=s_i, **kwargs) for s_i in s]), f


@pytest.mark.parametrize('criterion, n_comp, kwargs', [
    (criteria._max_normal, 6, dict(search_method='local')),
    (criteria._cs, 6, dict(s_af=300., tau_af=180., T=1.)),
    (criteria._EVMS, 3, {})])
def test_loop_over_points_warns_and_gives_same_result(criterion, n_comp, kwargs):
    es = FLife.EquivalentStress(input={'PSD': random_psd(n_comp, n_points=3, seed=1), 'f': F})
    ref, f = old_loop_over_points(es, criterion, **kwargs)
    with pytest.warns(DeprecationWarning, match='loop_over_points.*deprecated.*removed'):
        es.loop_over_points(criterion=criterion, **kwargs)
    np.testing.assert_allclose(es.eq_psd_multipoint[0], ref, rtol=1e-12)
    np.testing.assert_array_equal(es.eq_psd_multipoint[1], f)


PSD_METHODS = [
    ('max_normal', 6, {}), ('max_shear', 6, {}), ('max_normal_and_shear', 6, dict(s_af=300., tau_af=180.)),
    ('gao_mwcm', 6, dict(s_af=300., tau_af=180.)), ('cs', 6, dict(s_af=300., tau_af=180., T=1.)),
    ('EVMS', 6, {}), ('thermoelastic', 6, {}), ('Lemaitre', 6, dict(poisson_ratio=0.3)),
    ('Nieslony', 6, dict(s_af=300., tau_af=180.)), ('EVMS_out_of_phase', 3, {}),
    ('multiaxial_rainflow', 3, dict(k=5.))]


@pytest.mark.parametrize('name, n_comp, kwargs', PSD_METHODS)
def test_criteria_do_not_use_the_deprecated_loop(name, n_comp, kwargs):
    es = FLife.EquivalentStress(input={'PSD': random_psd(n_comp, n_points=2), 'f': F})
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        getattr(es, name)(**kwargs)
    assert es.eq_psd_multipoint[0].shape == (2, F.size)
