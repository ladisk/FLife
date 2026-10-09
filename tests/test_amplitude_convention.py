import numpy as np
import pytest
import FLife


def test_psd_trim_length_is_number_of_points():
    x = np.random.default_rng(0).normal(size=20000)
    sd = FLife.SpectralData(input={'time_history': x, 'dt': 1e-3}, nperseg=256, psd_trim_length=50)
    full = FLife.SpectralData(input={'time_history': x, 'dt': 1e-3}, nperseg=256)
    assert sd.psd.shape == (50, 2)
    np.testing.assert_array_equal(sd.psd, full.psd[:50])
