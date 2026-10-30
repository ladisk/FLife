"""LiWI and COIN-LiWI (amplitude spectrum input) against analytic references.

Schmidt AT, Pandiya N. Extension of the static equivalent stress hypotheses to linearly
vibrating systems using wave interference - The LiWI approach. Int J Fatigue 2021;147:106103.

Schmidt AT, Kraft J. A new equivalent stress approach based on complex invariants:
The COIN LiWI method. Int J Fatigue 2023;168:107474.
"""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

F = np.arange(0., 120., 2.)
DF = F[1] - F[0]
C, K = 1.8e22, 7.3


def amplitudes(n_points=None, n_comp=6, seed=0):
    """Random complex amplitude spectrum (f, n_comp) or (N, f, n_comp) under a smooth envelope."""
    rng = np.random.default_rng(seed)
    shape = (F.size, n_comp) if n_points is None else (n_points, F.size, n_comp)
    env = np.exp(-((F - 50) / 20)**2)[:, None] + 0.1
    return env * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def eq_psd(es):
    return es.eq_psd_multipoint[0] if es.multipoint else es.psd[:, 1]


# --- amplitude spectrum convention: PSD = |A|^2/df ---------------------------------------

def test_uniaxial_complex_amplitude_spectrum():
    a = amplitudes(n_comp=1)[:, 0]
    sd = FLife.SpectralData(input={'amplitude_spectrum': a, 'f': F})
    assert np.isrealobj(sd.psd)
    np.testing.assert_allclose(sd.psd[:, 1], np.abs(a)**2 / DF, rtol=1e-12)


@pytest.mark.parametrize('n_points', [None, 3])
@pytest.mark.parametrize('n_comp, method, args', [(3, 'liwi', {}), (6, 'coin_liwi', dict(k_a=1.7, k_phi=0.9))])
def test_uniaxial_stress_state_matches_uniaxial_path(n_points, n_comp, method, args):
    # sigma_x only: LiWI and COIN-LiWI give |sigma_x|, so the PSD equals the uniaxial path |A|^2/df
    amp = 3 * np.exp(-((F - 50) / 20)**2) + 0.5
    a = np.zeros((F.size, n_comp), complex)
    a[:, 0] = amp * np.exp(0.4j)
    if n_points:
        a = np.stack([a] * n_points)
    es = FLife.EquivalentStress(input={'amplitude_spectrum': a, 'f': F})
    getattr(es, method)(**args)
    psd = eq_psd(es)
    if n_points:
        psd = psd[-1]
        es.select_critical_point(n_points - 1)
    np.testing.assert_allclose(psd, amp**2 / DF, rtol=1e-12)
    uni = FLife.SpectralData(input={'amplitude_spectrum': amp, 'f': F})
    np.testing.assert_allclose(FLife.Dirlik(es).get_life(C, K), FLife.Dirlik(uni).get_life(C, K), rtol=1e-9)


@pytest.mark.parametrize('n_points', [None, 3])
def test_coin_liwi_rms_equals_evms_of_psd_matrix(n_points):
    # Schmidt and Kraft (2023), Eqs. 27 and 32: k_a^2 = 3, k_phi = 1 is the von Mises value of the
    # amplitudes, the same as EVMS (Preumont and Piefort 1994) of the PSD matrix S_ij = A_i conj(A_j)/df
    a = amplitudes(n_points)
    ea = FLife.EquivalentStress(input={'amplitude_spectrum': a, 'f': F})
    ea.coin_liwi(k_a=np.sqrt(3), k_phi=1.)
    ep = FLife.EquivalentStress(input={'PSD': a[..., :, None] * np.conj(a[..., None, :]) / DF, 'f': F})
    ep.EVMS()
    np.testing.assert_allclose(eq_psd(ea), np.real(eq_psd(ep)), rtol=1e-10)


@pytest.mark.parametrize('n_comp, method, args', [(3, 'liwi', {}), (6, 'coin_liwi', dict(k_a=1.7, k_phi=0.9))])
def test_single_point_equals_multipoint(n_comp, method, args):
    a = amplitudes(2, n_comp, seed=2)
    multi = FLife.EquivalentStress(input={'amplitude_spectrum': a, 'f': F})
    getattr(multi, method)(**args)
    for i in range(2):
        single = FLife.EquivalentStress(input={'amplitude_spectrum': a[i], 'f': F})
        getattr(single, method)(**args)
        np.testing.assert_allclose(single.psd[:, 1], multi.eq_psd_multipoint[0][i], rtol=1e-12)
        multi.select_critical_point(i)
        np.testing.assert_allclose(FLife.Dirlik(single).get_life(C, K), FLife.Dirlik(multi).get_life(C, K), rtol=1e-9)
