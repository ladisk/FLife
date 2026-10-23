import os
import warnings

import numpy as np
import pytest

import FLife


F = np.arange(0, 401.)


def band(f_lo, f_hi, level):
    psd = np.zeros_like(F)
    psd[(F >= f_lo) & (F <= f_hi)] = level
    return psd


def test_warns_when_eq24_conditions_fail():
    # Jun and Park (2020), Eq. (24) assumes D_2, D_3 and sigma_H are positive. On the test_basic
    # signal the Eq. (13) estimates of MRR give sigma_H < 0 and the PDF goes negative
    data = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'm1.npy'))
    t, x = data[::2], data[1::2]
    sd = FLife.SpectralData(input={'time_history': 100 * x / np.std(x), 'dt': t[1]}, nperseg=int(0.1 / t[1]))
    with pytest.warns(UserWarning, match=r'Eq\. \(24\)'):
        jp = FLife.JunPark(sd)
    assert jp.parameters['sigma_H'] < 0


def test_no_warning_when_eq24_conditions_hold():
    sd = FLife.SpectralData(input={'PSD': band(10, 20, 50.) + band(150, 170, 50.), 'f': F})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        jp = FLife.JunPark(sd)
    assert jp.parameters['sigma_H'] > 0
