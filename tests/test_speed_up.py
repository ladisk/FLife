"""Checks that speed-ups keep the previous results (synthetic PSDs only)."""
import numpy as np
import pytest
from scipy import stats
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.tools import pdf_rayleigh_sum
from FLife.freq_domain import zhao_baker

F = np.arange(0, 400.5, 0.25)


def band(f1, f2, var, f=F):
    """Flat PSD with variance `var` on [f1, f2]."""
    return np.where((f >= f1) & (f <= f2), var / (f2 - f1), 0.)


def _bimodal():
    return FLife.SpectralData(input={'PSD': band(10, 20, 50.) + band(150, 170, 50.), 'f': F})


def test_pdf_rayleigh_sum_ndtr_identical_to_norm_cdf():
    m0L, m0H = 3., 7.
    x = np.linspace(0, 15, 500)
    m0 = m0L + m0H
    L, H = m0L/m0, m0H/m0
    # reference: previous implementation with stats.norm.cdf
    ref = 1 / np.sqrt(m0) * (L * x/np.sqrt(m0) * np.exp(- (x/np.sqrt(m0))**2 /(2*L)) \
        + H * x/np.sqrt(m0) * np.exp(- (x/np.sqrt(m0))**2 /(2*H)) \
        + np.sqrt(2*np.pi*L*H) * ((x/np.sqrt(m0))**2 -1) * np.exp(-(x/np.sqrt(m0))**2/2) \
        * (stats.norm.cdf(np.sqrt(L/H)*x/np.sqrt(m0)) + stats.norm.cdf(np.sqrt(H/L)*x/np.sqrt(m0)) - 1))
    np.testing.assert_array_equal(pdf_rayleigh_sum(m0L, m0H)(x), ref)


# life values computed on origin/main (before the ndtr change)
@pytest.mark.parametrize('method, k, life', [
    (FLife.JiaoMoan, 3, 3214264.631922058),
    (FLife.JiaoMoan, 5, 8631.115431584436),
    (FLife.GaoMoan, 3, 2055967.097100138),
    (FLife.GaoMoan, 5, 4458.686242040772),
])
def test_jiao_moan_gao_moan_life_unchanged(method, k, life):
    assert method(_bimodal()).get_life(1e12, k) == pytest.approx(life, rel=1e-12)


def test_zhao_baker_method_2_integrate_pdf_unchanged():
    # value computed on origin/main
    zb = FLife.ZhaoBaker(_bimodal())
    with np.errstate(all='ignore'):
        life = zb.get_life(1e12, 3, method='method 2', integrate_pdf=True)
    assert life == pytest.approx(2718497.015730094, rel=1e-12)


def test_zhao_baker_method_2_solves_coefficients_once(monkeypatch):
    calls = []
    fsolve = zhao_baker.fsolve

    def counting_fsolve(*args, **kwargs):
        calls.append(1)
        return fsolve(*args, **kwargs)

    monkeypatch.setattr(zhao_baker, 'fsolve', counting_fsolve)
    zb = FLife.ZhaoBaker(_bimodal())
    with np.errstate(all='ignore'):
        zb.get_life(1e12, 3, method='method 2', integrate_pdf=True)
    assert len(calls) == 1
