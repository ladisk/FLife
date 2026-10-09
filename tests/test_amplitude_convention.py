import numpy as np
import pytest
import FLife

F = np.arange(0., 120., 2.)
DF = F[1] - F[0]
C, K = 1.8e22, 7.3


def amplitudes(n_points=None, n_comp=6, seed=0):
    rng = np.random.default_rng(seed)
    shape = (F.size, n_comp) if n_points is None else (n_points, F.size, n_comp)
    env = np.exp(-((F - 50) / 20)**2)[:, None]
    return env * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def psd_from_amplitude(a):
    return a[..., :, None] * np.conj(a[..., None, :]) / DF


def criterion_call(eq, name):
    if name == 'EVMS':
        eq.EVMS()
    else:
        eq.max_normal()


def eq_psd(eq):
    return eq.eq_psd_multipoint[0] if eq.multipoint else eq.psd[:, 1]


@pytest.mark.parametrize('name', ['EVMS', 'max_normal'])
@pytest.mark.parametrize('n_points', [None, 3])
@pytest.mark.parametrize('n_comp', [3, 6])
def test_amplitude_input_equals_psd_matrix_input(name, n_points, n_comp):
    a = amplitudes(n_points, n_comp)
    ea = FLife.EquivalentStress(input={'amplitude_spectrum': a, 'f': F})
    ep = FLife.EquivalentStress(input={'PSD': psd_from_amplitude(a), 'f': F})
    criterion_call(ea, name)
    criterion_call(ep, name)
    np.testing.assert_allclose(eq_psd(ea), eq_psd(ep), rtol=1e-9)
    if n_points:
        ea.select_critical_point(1)
        ep.select_critical_point(1)
    np.testing.assert_allclose(FLife.Dirlik(ea).get_life(C, K), FLife.Dirlik(ep).get_life(C, K), rtol=1e-9)


@pytest.mark.parametrize('n_comp', [3, 6])
def test_uniaxial_stress_state_matches_uniaxial_path(n_comp):
    amp = 3 * np.exp(-((F - 50) / 20)**2) + 0.5
    a = np.zeros((F.size, n_comp), complex)
    a[:, 0] = amp
    uni = FLife.SpectralData(input={'amplitude_spectrum': amp, 'f': F})
    ref = FLife.Dirlik(uni).get_life(C, K)
    for name in ('EVMS', 'max_normal'):
        eq = FLife.EquivalentStress(input={'amplitude_spectrum': a, 'f': F})
        criterion_call(eq, name)
        np.testing.assert_allclose(FLife.Dirlik(eq).get_life(C, K), ref, rtol=1e-6)


@pytest.mark.parametrize('n_comp, method', [(3, 'liwi'), (6, 'coin_liwi')])
def test_liwi_single_point_equals_multipoint(n_comp, method):
    a = amplitudes(2, n_comp, seed=2)
    args = {} if method == 'liwi' else dict(k_a=1.7, k_phi=0.9)
    multi = FLife.EquivalentStress(input={'amplitude_spectrum': a, 'f': F})
    getattr(multi, method)(**args)
    for i in range(2):
        single = FLife.EquivalentStress(input={'amplitude_spectrum': a[i], 'f': F})
        getattr(single, method)(**args)
        np.testing.assert_allclose(single.psd[:, 1], multi.eq_psd_multipoint[0][i], rtol=1e-12)
        multi.select_critical_point(i)
        np.testing.assert_allclose(FLife.Dirlik(single).get_life(C, K), FLife.Dirlik(multi).get_life(C, K), rtol=1e-9)


def test_liwi_uniaxial_amplitude_is_a_squared_over_df():
    amp = np.full(F.size, 2.)
    a = np.zeros((F.size, 3), complex)
    a[:, 0] = amp
    eq = FLife.EquivalentStress(input={'amplitude_spectrum': a, 'f': F})
    eq.liwi()
    np.testing.assert_allclose(eq.psd[:, 1], amp**2 / DF, rtol=1e-9)


def test_psd_trim_length_is_number_of_points():
    x = np.random.default_rng(0).normal(size=20000)
    sd = FLife.SpectralData(input={'time_history': x, 'dt': 1e-3}, nperseg=256, psd_trim_length=50)
    full = FLife.SpectralData(input={'time_history': x, 'dt': 1e-3}, nperseg=256)
    assert sd.psd.shape == (50, 2)
    np.testing.assert_array_equal(sd.psd, full.psd[:50])
