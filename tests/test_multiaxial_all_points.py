"""Criteria evaluated for all points at once give the same equivalent PSD as before (one point at a time).

Reference values were computed with FLife 2.x, where these criteria looped over points and frequencies.
"""
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
FREQ_DATA = np.arange(0, 240, 3)


def random_psd(n_comp, n_points=None, seed=0):
    """Hermitian PSD matrix of two coherent sources with complex mode shapes, (f, n, n) or (N, f, n, n)."""
    rng = np.random.default_rng(seed)
    shape = (2, n_comp) if n_points is None else (n_points, 2, n_comp)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    P2 = np.roll(P, 30)
    return (np.einsum('...i,...j,f->...fij', a[..., 0, :], a[..., 0, :].conj(), P)
            + np.einsum('...i,...j,f->...fij', a[..., 1, :], a[..., 1, :].conj(), P2))


def data_psd(dim):
    return np.load(os.path.join(my_path, '..', 'data', f'test_multiaxial_PSD_{dim}.npy'))[331:335]


def eq_psd(S, f, name, **kwargs):
    es = FLife.EquivalentStress(input={'PSD': S, 'f': f})
    getattr(es, name)(**kwargs)
    return es.eq_psd_multipoint[0]


# --- Nieslony and EVMS_out_of_phase (Bonte) ----------------------------------------------

PSD_CRITERIA = [('Nieslony', 3, dict(s_af=300., tau_af=180., coefficient_load_type='tension')),
                ('Nieslony', 6, dict(s_af=300., tau_af=180., coefficient_load_type='torsion')),
                ('EVMS_out_of_phase', 3, {})]


@pytest.mark.parametrize('name, n_comp, kwargs', PSD_CRITERIA)
def test_psd_criterion_accepts_all_points_at_once(name, n_comp, kwargs):
    S = random_psd(n_comp, n_points=4, seed=3)
    criterion = getattr(criteria, '_' + name)
    all_points = criterion(None, S, **kwargs)
    assert all_points.shape == S.shape[:2]
    for i in range(len(S)):
        np.testing.assert_allclose(all_points[i], criterion(None, S[i], **kwargs), rtol=1e-12)


@pytest.mark.parametrize('dim', ['2D', '3D'])
@pytest.mark.parametrize('load_type, sums, value', [
    ('tension', [5634.316285048235, 5311.034763902596, 4900.717552512217, 4417.6312203350035], 14.98841443136467),
    ('torsion', [2028.353862617364, 1911.9725150049344, 1764.258318904398, 1590.3472393206011], 5.395829195291281)])
def test_nieslony_reference_data(dim, load_type, sums, value):
    s_eq = eq_psd(data_psd(dim), FREQ_DATA, 'Nieslony', s_af=300., tau_af=180., coefficient_load_type=load_type)
    np.testing.assert_allclose(s_eq.sum(axis=1), sums, rtol=1e-12)
    np.testing.assert_allclose(s_eq[2, 30], value, rtol=1e-12)


@pytest.mark.parametrize('n_comp, sums, value', [
    (3, [117.85523640077524, 361.9516037201301, 293.1316331526373], 5.109337693599989),
    (6, [924.1492389597046, 1043.8131821212362, 441.9928057542803], 13.533477558899634)])
def test_nieslony_reference_random(n_comp, sums, value):
    s_eq = eq_psd(random_psd(n_comp, n_points=3, seed=5), F, 'Nieslony', s_af=300., tau_af=180.)
    np.testing.assert_allclose(s_eq.sum(axis=1), sums, rtol=1e-12)
    np.testing.assert_allclose(s_eq[1, 60], value, rtol=1e-12)


def test_bonte_reference():
    s_eq = eq_psd(data_psd('2D'), FREQ_DATA, 'EVMS_out_of_phase')
    np.testing.assert_allclose(s_eq.sum(axis=1), [5765.911460753015, 5445.473962386703, 5034.3943973267305, 4547.138532397865], rtol=1e-12)
    np.testing.assert_allclose(s_eq[2, 30], 15.500192425609512, rtol=1e-12)
    s_eq = eq_psd(random_psd(3, n_points=3, seed=5), F, 'EVMS_out_of_phase')
    np.testing.assert_allclose(s_eq.sum(axis=1), [88.23349585446338, 353.03078120537305, 265.55326885678875], rtol=1e-12)
    np.testing.assert_allclose(s_eq[1, 60], 5.165778657862327, rtol=1e-12)


@pytest.mark.parametrize('n_points', [None, 2])
def test_bonte_rejects_3d_stress(n_points):
    es = FLife.EquivalentStress(input={'PSD': random_psd(6, n_points), 'f': F})
    with pytest.raises(ValueError):
        es.EVMS_out_of_phase()
