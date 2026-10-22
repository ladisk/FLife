import numpy as np
import pytest
from scipy.integrate import quad

import FLife


F = np.arange(0, 401.)


def band(f_lo, f_hi, level):
    psd = np.zeros_like(F)
    psd[(F >= f_lo) & (F <= f_hi)] = level
    return psd


SPECTRA = {
    'bimodal': band(10, 20, 50.) + band(150, 170, 50.),
    'wideband': band(10, 300, 1.),
}

# (class, keyword arguments of get_PDF and get_life)
METHODS = {
    'Narrowband': (FLife.Narrowband, {}),
    'Tovo Benasciutti 1': (FLife.TovoBenasciutti, {'method': 'method 1'}),
    'Tovo Benasciutti 2': (FLife.TovoBenasciutti, {'method': 'method 2'}),
    'Tovo Benasciutti 3': (FLife.TovoBenasciutti, {'method': 'method 3'}),
    'Dirlik': (FLife.Dirlik, {}),
    'Zhao Baker 1': (FLife.ZhaoBaker, {'method': 'method 1'}),
    'Zhao Baker 2': (FLife.ZhaoBaker, {'method': 'method 2'}),
    'Park': (FLife.Park, {}),
    'Jun Park': (FLife.JunPark, {}),
}


@pytest.mark.parametrize('spectrum', SPECTRA)
@pytest.mark.parametrize('name', METHODS)
def test_pdf_integrates_to_one(name, spectrum):
    sd = FLife.SpectralData(input={'PSD': SPECTRA[spectrum], 'f': F})
    cls, kw = METHODS[name]
    m = cls(sd)
    area = quad(lambda s: m.get_PDF(s, **kw), 0, np.inf)[0]
    # Jun and Park (2020), Eq. (17): the PDF is scaled by the correction factor Qc
    expected = m._get_Qc() if name == 'Jun Park' else 1.
    np.testing.assert_allclose(area, expected, rtol=1e-5)


@pytest.mark.parametrize('k', [3., 6.])
@pytest.mark.parametrize('spectrum', SPECTRA)
@pytest.mark.parametrize('name', METHODS)
def test_pdf_damage_equals_closed_form(name, spectrum, k):
    sd = FLife.SpectralData(input={'PSD': SPECTRA[spectrum], 'f': F})
    cls, kw = METHODS[name]
    m = cls(sd)
    C = 1e12
    np.testing.assert_allclose(m.get_life(C, k, integrate_pdf=True, **kw), m.get_life(C, k, **kw), rtol=1e-5)
