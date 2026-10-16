"""Checks of fixed bugs against analytic references (synthetic PSDs only)."""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

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
