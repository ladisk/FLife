"""EquivalentStress passes T, fs and rg on to the time history of the equivalent stress."""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

F = np.arange(0, 201, 1.)
P = np.exp(-0.5 * ((F - 50) / 8)**2)
P[0] = 0.
T, FS = 2., 2000.


def psd(n_points=None):
    a = np.array([1., 0.5, 0.3]) if n_points is None else np.random.default_rng(0).normal(size=(n_points, 3))
    return np.einsum('...i,...j,f->...fij', a, a, P)


def eq_data(S, seed, **kwargs):
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F}, T=T, fs=FS, rg=np.random.default_rng(seed), **kwargs)
    es.EVMS()
    if es.multipoint:
        es.select_critical_point(1)
    return es.data


@pytest.mark.parametrize('n_points', [None, 3])
def test_same_seed_gives_same_time_history(n_points):
    S = psd(n_points)
    x = eq_data(S, seed=4)
    assert x.size == T * FS
    np.testing.assert_array_equal(x, eq_data(S, seed=4))
    assert not np.allclose(x, eq_data(S, seed=5))


def test_spectral_data_input_keeps_seed():
    def data(seed):
        sd = FLife.SpectralData(input={'PSD': psd(), 'f': F}, T=T, fs=FS, rg=np.random.default_rng(seed))
        es = FLife.EquivalentStress(sd)
        es.EVMS()
        return es.data
    np.testing.assert_array_equal(data(4), data(4))
    np.testing.assert_array_equal(data(4), eq_data(psd(), seed=4))


def test_sampling_frequency_checked_at_input():
    # same check as SpectralData: fs >= 2*fmax
    with pytest.raises(ValueError):
        FLife.EquivalentStress(input={'PSD': psd(), 'f': F}, T=T, fs=300.)
