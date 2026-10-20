import numpy as np
from scipy.integrate import trapezoid

import FLife


def _flat_band(n_zeros_after):
    """Flat PSD between 100 and 200 Hz, followed by `n_zeros_after` zero points."""
    f = np.arange(0, 201. + n_zeros_after)
    psd = np.zeros_like(f)
    psd[100:201] = 1.
    return f, psd


def _full_integral(f, psd, moments=(0, 2, 4)):
    w = 2 * np.pi * f
    return [trapezoid(w**i * psd, f) for i in moments]


def test_band_moments_add_up_to_full_integral():
    """The bands together must cover the whole PSD, including the drop to zero after the last nonzero point."""
    f, psd = _flat_band(n_zeros_after=100)
    sd = FLife.SpectralData(input={'PSD': psd, 'f': f})
    for N in (1, 2, 3):
        m = sd.get_spectral_moments(('equalAreaBands', N), moments=[0, 2, 4])
        np.testing.assert_allclose(np.sum(m, axis=0), _full_integral(f, psd), rtol=1e-12)


def test_single_line_psd():
    f = np.arange(0, 301.)
    psd = np.zeros_like(f)
    psd[150] = 1.
    sd = FLife.SpectralData(input={'PSD': psd, 'f': f})
    np.testing.assert_allclose(sd.moments[[0, 2, 4]], _full_integral(f, psd), rtol=1e-12)
