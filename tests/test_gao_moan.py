from math import comb

import numpy as np
from scipy.special import gamma

import FLife


F = np.arange(0, 401.)


def band(f_lo, f_hi, level):
    psd = np.zeros_like(F)
    psd[(F >= f_lo) & (F <= f_hi)] = level
    return psd


def rayleigh_moment(n, m0):
    """E[R**n] of a Rayleigh amplitude with variance m0 of the underlying process."""
    return (2 * m0)**(n / 2) * gamma(1 + n / 2)


def test_trimodal_large_cycle_moments():
    # Gao and Moan (2008), Table 1 and Eqs. (7)-(8): the large-cycle amplitude is R_H + R_M + R_L,
    # a sum of independent Rayleigh amplitudes, so E[S**k] is the multinomial sum of Rayleigh moments
    sd = FLife.SpectralData(input={'PSD': band(10, 300, 100.), 'f': F})
    gm = FLife.GaoMoan(sd)
    m0_LF, m0_MF, m0_HF, nu_L, C = 3., 2., 0.5, 1.7, 1e3
    for k in (0, 3, 5, 7):
        expected = sum(comb(k, i) * comb(k - i, j) * rayleigh_moment(i, m0_LF) * rayleigh_moment(j, m0_MF)
                       * rayleigh_moment(k - i - j, m0_HF) for i in range(k + 1) for j in range(k - i + 1))
        d = gm._damage_intesity_trimodal_LF(m0_LF=m0_LF, m0_MF=m0_MF, m0_HF=m0_HF, nu_L=nu_L, C=C, k=k)
        np.testing.assert_allclose(d, nu_L * expected / C, rtol=1e-8, err_msg=f'k={k}')
