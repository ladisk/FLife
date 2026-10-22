import numpy as np
import pytest
from scipy.integrate import quad

import FLife


F = np.arange(0, 401.)


def band(f_lo, f_hi, level):
    psd = np.zeros_like(F)
    psd[(F >= f_lo) & (F <= f_hi)] = level
    return psd


@pytest.mark.parametrize('method', ['method 1', 'method 2', 'method 3'])
def test_pdf_damage_equals_closed_form(method):
    # Benasciutti and Tovo (2005), Eq. (28) gives the amplitude PDF, Eq. (16) its damage
    # and Eq. (27) the same damage in closed form
    sd = FLife.SpectralData(input={'PSD': band(10, 20, 50.) + band(150, 170, 50.), 'f': F})
    tb = FLife.TovoBenasciutti(sd)
    C, k = 1e12, 5.
    np.testing.assert_allclose(tb.get_life(C, k, method=method, integrate_pdf=True),
                               tb.get_life(C, k, method=method), rtol=1e-6)


def test_pdf_mass():
    # Eq. (19): level crossing counting gives alpha2 cycles per peak,
    # so the Eq. (28) PDF integrates to b alpha2 + 1 - b
    sd = FLife.SpectralData(input={'PSD': band(10, 20, 50.) + band(150, 170, 50.), 'f': F})
    tb = FLife.TovoBenasciutti(sd)
    b, alpha2 = tb._calculate_coefficient('method 2'), sd.alpha2
    np.testing.assert_allclose(quad(tb.get_PDF, 0, np.inf)[0], 1 - b * (1 - alpha2), rtol=1e-6)
