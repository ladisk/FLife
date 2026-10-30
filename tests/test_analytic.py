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


def test_low_warns_on_negative_large_cycle_damage():
    # narrowband PSD at k=5: the sixth-order series of the large cycles (Low 2010, App. C) gives negative damage
    sd = FLife.SpectralData(input={'PSD': band(50, 52, 10.), 'f': F})
    with pytest.warns(UserWarning, match='Large-cycle damage is negative'):
        FLife.Low(sd).get_life(1e18, 5)


def test_low_no_warning_for_well_separated_bimodal():
    # beta about 10, studied range of Low (2010)
    sd = FLife.SpectralData(input={'PSD': band(10, 20, 50.) + band(150, 170, 50.), 'f': F})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        FLife.Low(sd, ('userDefinedBands', [80, 400])).get_life(1e18, 3)


def test_low_warns_for_small_beta():
    # centre frequencies 30 and 90 Hz, beta about 3 (Low 2010 studied beta = 4 and 10)
    sd = FLife.SpectralData(input={'PSD': band(25, 35, 50.) + band(85, 95, 50.), 'f': F})
    with pytest.warns(UserWarning, match='studied the method only for beta = 4 and 10'):
        FLife.Low(sd, ('userDefinedBands', [60, 400])).get_life(1e12, 3)


def _coherent_biaxial_psd(G, phi, f):
    """Rank-1 (f, 3, 3) PSD matrix, S[j, k] = conj(a_j) a_k, a_j = sqrt(G_j) exp(i phi_j)."""
    a = np.sqrt(G) * np.exp(1j*np.asarray(phi))
    S = np.outer(np.conj(a), a)
    return np.repeat(S[np.newaxis], len(f), axis=0)


def _max_von_mises_squared(G, phi):
    """Time-domain max of sigma_vM^2 for harmonic components (Bonte et al. 2007, Eqs. 4 and 9)."""
    wt = np.linspace(0, 2*np.pi, 20001)
    sx, sy, txy = (np.sqrt(G[j]) * np.cos(wt + phi[j]) for j in range(3))
    return np.max(sx**2 + sy**2 - sx*sy + 3*txy**2)


@pytest.mark.parametrize('G, phi', [
    ([0., 4., 1.], [0., 0., np.pi/2]),        # sigma_x = 0, sigma_y and tau_xy 90 deg apart
    ([0., 1., 2.], [0., 0.3, 1.9]),            # sigma_x = 0
    ([1e-30, 2., 1.], [0., -0.7, 2.2]),        # sigma_x negligible
    ([1., 3., 0.5], [0.4, -1.1, 2.0]),         # sigma_x not the largest component
    ([3., 1., 0.5], [0.4, -1.1, 2.0]),         # sigma_x the largest component
])
def test_evms_out_of_phase_matches_time_domain_maximum(G, phi):
    f = np.arange(1., 11.)
    es = FLife.EquivalentStress(input={'PSD': _coherent_biaxial_psd(G, phi, f), 'f': f})
    es.EVMS_out_of_phase()
    np.testing.assert_allclose(es.psd[:, 1], _max_von_mises_squared(G, phi), rtol=1e-6)


def test_evms_out_of_phase_swap_x_y():
    # the von Mises stress is symmetric in sigma_x and sigma_y
    f = np.arange(1., 11.)
    G, phi = np.array([0., 2., 1.]), np.array([0., 0.5, 2.1])
    res = []
    for order in ([0, 1, 2], [1, 0, 2]):
        es = FLife.EquivalentStress(input={'PSD': _coherent_biaxial_psd(G[order], phi[order], f), 'f': f})
        es.EVMS_out_of_phase()
        res.append(es.psd[:, 1])
    np.testing.assert_allclose(res[0], res[1], rtol=1e-12)
