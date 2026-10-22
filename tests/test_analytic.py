"""Checks of fixed bugs against analytic references (synthetic PSDs only)."""
import warnings
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.tools import random_gaussian

F = np.arange(0, 400.5, 0.25)


def band(f1, f2, var, f=F):
    """Flat PSD with variance `var` on [f1, f2]."""
    return np.where((f >= f1) & (f <= f2), var / (f2 - f1), 0.)


def _sine_spectral_data(mean, amplitude=1., n_periods=200, fs=1000., f0=10.):
    t = np.arange(int(n_periods*fs/f0)) / fs
    x = mean + amplitude*np.sin(2*np.pi*f0*t)
    return FLife.SpectralData(input={'time_history': x, 'dt': 1/fs})


def test_jiao_moan_gao_moan_single_band():
    sd = FLife.SpectralData(input={'PSD': band(10, 300, 100.), 'f': F})
    C, k = 1e12, 5.
    nb = FLife.Narrowband(sd).get_life(C, k)
    np.testing.assert_allclose(FLife.JiaoMoan(sd, ('equalAreaBands', 1)).get_life(C, k), nb, rtol=1e-12)
    np.testing.assert_allclose(FLife.GaoMoan(sd, ('equalAreaBands', 1)).get_life(C, k), nb, rtol=1e-12)


def test_gao_moan_bimodal_runs():
    sd = FLife.SpectralData(input={'PSD': band(10, 20, 50.) + band(150, 170, 50.), 'f': F})
    life = FLife.GaoMoan(sd, ('equalAreaBands', 2)).get_life(1e12, 5.)
    assert np.isfinite(life) and life > 0


def test_jun_park_validity_warning():
    # narrow band: sqrt(1 - alpha1**2) < 0.3 is outside the validated range
    sd_nb = FLife.SpectralData(input={'PSD': band(95, 105, 100.), 'f': F})
    with pytest.warns(UserWarning, match='Qc'):
        FLife.JunPark(sd_nb).get_life(1e12, 5.)

    import warnings
    sd_bb = FLife.SpectralData(input={'PSD': band(10, 20, 50.) + band(150, 170, 50.), 'f': F})
    assert np.sqrt(1 - sd_bb.alpha1**2) > 0.3
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        FLife.JunPark(sd_bb).get_life(1e12, 5.)


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


def test_psd_trim_length_is_number_of_points():
    t = np.arange(20000) / 1000.
    x = np.random.default_rng(0).standard_normal(t.size)
    full = FLife.SpectralData(input={'time_history': x, 'dt': 1e-3}, nperseg=500)
    trimmed = FLife.SpectralData(input={'time_history': x, 'dt': 1e-3}, nperseg=500, psd_trim_length=100)
    assert trimmed.psd.shape == (100, 2)
    np.testing.assert_array_equal(trimmed.psd, full.psd[:100])


# --- performance rewrites keep the old results -------------------------------------------

# Low.get_life computed on origin/main with the previous scipy.integrate.dblquad implementation
# (C=1e18, same band splitting).
LOW_DBLQUAD = {
    'narrowband': (band(50, 52, 10.), {2: 910399463675059.5, 3: 187333090987999.44}),
    'bimodal': (band(10, 20, 50.) + band(150, 170, 50.), {3: 3502769111330.4214, 5: 8487241299.902009}),
    'broadband': (band(10, 300, 100.), {3: 1587775396130.4294, 5: 3203117492.3381845}),
}


@pytest.mark.parametrize('name', list(LOW_DBLQUAD))
def test_low_gauss_legendre_matches_dblquad(name):
    psd, expected = LOW_DBLQUAD[name]
    sd = FLife.SpectralData(input={'PSD': psd, 'f': F})
    for k, life in expected.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert FLife.Low(sd).get_life(1e18, k) == pytest.approx(life, rel=1e-8)


def test_bands_method_modified_psd_matches_loop():
    sd = FLife.SpectralData(input={'PSD': band(10, 300, 100.), 'f': F})
    bm = FLife.BandsMethod(sd, PSD_splitting=('equalAreaBands', 1))
    f_ref, k = 80., 3
    loop = np.array([(fi / f_ref)**(2/k) * p for fi, p in zip(sd.psd[:, 0], sd.psd[:, 1])])
    assert bm._m0_ref_modifiedPSD(k, f_ref) == pytest.approx(np.trapz(loop, dx=F[1] - F[0]) if hasattr(np, 'trapz') else np.trapezoid(loop, dx=F[1] - F[0]), rel=1e-14)


def test_cached_coefficients_match_uncached_moments():
    sd = FLife.SpectralData(input={'PSD': band(10, 20, 50.) + band(150, 170, 50.), 'f': F})
    assert sd._moment_cache is None  # cache only lives during construction
    np.testing.assert_array_equal(sd.moments, np.ravel(sd.get_spectral_moments(sd.PSD_splitting)))
    m = sd.moments
    assert sd.nu == pytest.approx(np.sqrt(m[2] / m[0]) / (2*np.pi), rel=1e-14)
    assert sd.alpha2 == pytest.approx(m[2] / np.sqrt(m[0] * m[4]), rel=1e-14)
    assert sd.alpha1 == pytest.approx(m[1] / np.sqrt(m[0] * m[2]), rel=1e-14)
    # and after construction new splittings are computed from the current PSD
    sd.psd[:, 1] *= 2
    assert sd.get_spectral_moments(sd.PSD_splitting, moments=[0])[0, 0] == pytest.approx(2 * m[0], rel=1e-12)


def test_random_amplitude_same_seed_as_list_comprehension():
    psd = band(10, 100, 4.)
    new = random_gaussian(F, psd, T=20, fs=1000, rg=np.random.default_rng(5), random_amplitude=True)[1]
    # reference: previous implementation drew the Rayleigh amplitudes one by one
    rg = np.random.default_rng(5)
    N = 20 * 1000
    freq_new = np.arange(0, N // 2 + 1) / N * 1000
    ampl = np.sqrt(np.interp(freq_new, F, psd, left=0, right=0) * N * 1000 / 2)
    phase = rg.uniform(0, 1, len(freq_new))
    ampl = np.array([rg.rayleigh(size=1, scale=x)[0] for x in ampl])
    ref = np.fft.irfft(ampl * np.exp(2j * np.pi * phase), n=N)
    np.testing.assert_allclose(new, ref / np.std(ref) * np.std(new), rtol=1e-12, atol=1e-12)
