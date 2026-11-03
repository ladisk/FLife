"""Every criterion gives the same equivalent PSD for one point as for that point in a multi-point input."""
import warnings
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

F = np.arange(0, 240, 3.)
F[0] = 1e-3


def data_psd(dim):
    return np.load(os.path.join(my_path, '..', 'data', f'test_multiaxial_PSD_{dim}.npy'))[331:334]


@pytest.mark.parametrize('name, dim, kwargs', [
    ('max_normal', '3D', {}), ('max_shear', '3D', {}), ('max_normal_and_shear', '3D', dict(s_af=300., tau_af=180.)),
    ('gao_mwcm', '3D', dict(s_af=300., tau_af=180.)), ('cs', '3D', dict(s_af=300., tau_af=180., T=1.)),
    ('EVMS', '3D', {}), ('thermoelastic', '3D', {}), ('Lemaitre', '3D', dict(poisson_ratio=0.3)),
    ('Nieslony', '3D', dict(s_af=300., tau_af=180.)), ('EVMS_out_of_phase', '2D', {}),
    ('multiaxial_rainflow', '2D', dict(k=5.)), ('max_normal', '2D', {}), ('EVMS', '2D', {})])
def test_single_point_matches_multi_point(name, dim, kwargs):
    S = data_psd(dim)
    multi = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    getattr(multi, name)(**kwargs)
    for i in range(len(S)):
        single = FLife.EquivalentStress(input={'PSD': S[i], 'f': F})
        getattr(single, name)(**kwargs)
        np.testing.assert_array_equal(single.psd[:, 0], F)
        np.testing.assert_allclose(single.psd[:, 1], multi.eq_psd_multipoint[0][i], rtol=1e-12, atol=0)
        assert single.moments[0] > 0
