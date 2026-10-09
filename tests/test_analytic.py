"""Fast checks of core and uniaxial code against analytic references (synthetic PSDs only)."""
import numpy as np
import pytest
import warnings
from scipy.special import gamma, comb
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.tools import random_gaussian

F = np.arange(0, 400.5, 0.25)


def band(f1, f2, var, f=F):
    """Flat PSD with variance `var` on [f1, f2]."""
    return np.where((f >= f1) & (f <= f2), var / (f2 - f1), 0.)


def rayleigh_moment(n, m0):
    """E[X**n] of a Rayleigh amplitude with variance parameter m0."""
    return (2*m0)**(n/2) * gamma(1 + n/2)


def flat_band_moments(S0, f1, f2, moments):
    return np.array([S0 * (2*np.pi)**i * (f2**(i+1) - f1**(i+1)) / (i+1) for i in moments])


def multiaxial_psd(f, psd_x, n_points=None):
    """Uniaxial stress in x as a 3D Voigt PSD matrix, shape (f,6,6) or (N,f,6,6)."""
    s = np.zeros((len(f), 6, 6))
    s[:, 0, 0] = psd_x
    if n_points is not None:
        s = np.repeat(s[None], n_points, axis=0)
    return s


# --- spectral moments and band splitting ----------------------------------------------

def test_equal_area_band_moments_analytic():
    # band edges half way between samples: the sampled PSD has the same area as the flat band
    df, S0, f1, f2 = 0.01, 2.0, 50.005, 150.005
    f = np.arange(0, 200, df)
    psd = np.where((f > f1) & (f < f2), S0, 0.)
    sd = FLife.SpectralData(input={'PSD': psd, 'f': f})

    analytic = flat_band_moments(S0, f1, f2, range(5))
    np.testing.assert_allclose(sd.moments, analytic, rtol=1e-6)

    for N in (2, 3):
        m = sd.get_spectral_moments(('equalAreaBands', N), moments=[0, 2, 4])
        # bands share their edge samples, so band moments add up to the full moments
        np.testing.assert_allclose(m.sum(axis=0), sd.moments[[0, 2, 4]], rtol=1e-12)
        np.testing.assert_allclose(m[:, 0], analytic[0]/N, atol=S0*df)

    assert sd.get_spectral_moments(('equalAreaBands', np.int64(2))).shape == (2, 5)


def test_single_moment_equals_bands_method():
    sd = FLife.SpectralData(input={'PSD': band(10, 300, 100.), 'f': F})
    C, k = 1e12, 5.
    np.testing.assert_allclose(FLife.SingleMoment(sd).get_life(C, k),
                               FLife.BandsMethod(sd).get_life(C, k), rtol=1e-10)


# --- time history generation ----------------------------------------------------------

def test_time_history_reproducible():
    psd = band(50, 150, 10.)
    a = FLife.SpectralData(input={'PSD': psd, 'f': F}, T=1, fs=2000, rg=np.random.default_rng(1))
    b = FLife.SpectralData(input={'PSD': psd, 'f': F}, T=1, fs=2000, rg=np.random.default_rng(1))
    np.testing.assert_array_equal(a.data, b.data)


def test_equivalent_stress_forwards_T_fs_rg():
    s = multiaxial_psd(F, band(50, 150, 10.))
    runs = []
    for _ in range(2):
        eq = FLife.EquivalentStress(input={'PSD': s, 'f': F}, T=1, fs=2000, rg=np.random.default_rng(3))
        assert eq.t == 1 and eq.fs == 2000
        eq.set_eq_stress(eq_psd=s[:, 0, 0], f=F)
        runs.append(eq.data)
    np.testing.assert_array_equal(runs[0], runs[1])

    with pytest.raises(ValueError, match='fs'):
        FLife.EquivalentStress(input={'PSD': s, 'f': F}, T=1, fs=200)


# --- input validation -----------------------------------------------------------------

def test_multiaxial_shape_and_fs_checks():
    s = multiaxial_psd(F, band(50, 150, 10.))
    with pytest.raises(Exception, match='Input Error'):
        FLife.SpectralData(input={'PSD': s[:, :, :3], 'f': F})  # (f,6,3)
    with pytest.raises(Exception, match='Input Error'):
        FLife.SpectralData(input={'amplitude_spectrum': np.ones((len(F), 4)), 'f': F})  # (f,4)

    # fs < 2*fmax, single point and multipoint (high frequency content at the last point only)
    with pytest.raises(ValueError, match='fs'):
        FLife.SpectralData(input={'PSD': s, 'f': F}, T=1, fs=250)
    sm = multiaxial_psd(F, band(10, 20, 10.), n_points=3)
    sm[-1, :, 1, 1] = band(50, 150, 1.)
    with pytest.raises(ValueError, match='fs'):
        FLife.SpectralData(input={'PSD': sm, 'f': F}, T=1, fs=250)
    sd = FLife.SpectralData(input={'PSD': sm, 'f': F}, T=1, fs=2000)
    assert sd.multipoint


def test_input_validation():
    psd = band(50, 150, 10.)
    bad_inputs = {
        'unsorted f': {'PSD': psd[::-1], 'f': F[::-1]},
        'two-sided f': {'PSD': np.r_[psd[:0:-1], psd], 'f': np.r_[-F[:0:-1], F]},
        'negative PSD': {'PSD': psd - 1., 'f': F},
        'zero PSD': {'PSD': np.zeros_like(F), 'f': F},
        'length mismatch': {'PSD': psd[:-1], 'f': F},
    }
    for name, input_dict in bad_inputs.items():
        with pytest.raises(ValueError):
            FLife.SpectralData(input=input_dict)

    # round-off negative values are accepted
    FLife.SpectralData(input={'PSD': psd - 1e-15, 'f': F})

    # multiaxial: complex cross terms are fine, a non-uniform grid is not
    s = multiaxial_psd(F, psd).astype(complex)
    s[:, 0, 1] = 1j*psd
    s[:, 1, 0] = -1j*psd
    FLife.SpectralData(input={'PSD': s, 'f': F})
    f_nonuniform = F.copy()
    f_nonuniform[100:] += 0.1
    with pytest.raises(ValueError, match='uniform'):
        FLife.SpectralData(input={'PSD': s, 'f': f_nonuniform})


def test_ndarray_input_raises_clear_error():
    with pytest.raises(ValueError, match='Unrecognized Input'):
        FLife.SpectralData(input=np.ones(100))


# --- rainflow -------------------------------------------------------------------------

def _sine_spectral_data(mean, amplitude=1., n_periods=200, fs=1000., f0=10.):
    t = np.arange(int(n_periods*fs/f0)) / fs
    x = mean + amplitude*np.sin(2*np.pi*f0*t)
    return FLife.SpectralData(input={'time_history': x, 'dt': 1/fs})


def test_rainflow_requires_T_and_fs():
    sd = FLife.SpectralData(input={'PSD': band(50, 150, 10.), 'f': F})
    with pytest.raises(Exception, match='T and fs'):
        FLife.Rainflow(sd, fs=2000)
    rf = FLife.Rainflow(sd, T=1, fs=2000, rg=np.random.default_rng(0))
    assert rf.spectral_data.data.size == 2000


def test_rainflow_nr_load_classes():
    import fatpack
    sd = FLife.SpectralData(input={'PSD': band(50, 150, 10.), 'f': F}, T=2, fs=2000, rg=np.random.default_rng(0))
    C, k = 1e12, 5.
    for nr in (8, 512):
        ranges = fatpack.find_rainflow_ranges(sd.data, k=nr)
        expected = sd.t / np.sum((ranges/2)**k / C)
        np.testing.assert_allclose(FLife.Rainflow(sd).get_life(C, k, nr_load_classes=nr), expected, rtol=1e-12)


def test_rainflow_goodman():
    C, k = 1e12, 5.
    sd = _sine_spectral_data(mean=2.)
    rf = FLife.Rainflow(sd)
    life = rf.get_life(C, k)
    assert rf.get_life(C, k, Su=False) == life  # old default still means "off"
    np.testing.assert_allclose(rf.get_life(C, k, Su=10.), life * (1 - 2./10.)**k, rtol=1e-6)
    for Su in (0., -10.):
        with pytest.raises(ValueError):
            rf.get_life(C, k, Su=Su)
    with pytest.raises(ValueError, match='Goodman'):
        rf.get_life(C, k, Su=1.5)
    neg = FLife.Rainflow(_sine_spectral_data(mean=-2.))
    np.testing.assert_allclose(neg.get_life(C, k, Su=10.), neg.get_life(C, k) * (1 + 2./10.)**k, rtol=1e-6)
    assert np.isfinite(neg.get_life(C, k, Su=1.5))
