import os
import sys
import numpy as np
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

F = np.arange(0, 100.5, 0.5)
PSD = np.where((F > 10) & (F < 50), 1.0, 0.0)


def psd_input(f=F, psd=PSD):
    return {'PSD': np.array(psd, dtype=float), 'f': np.array(f, dtype=float)}


def test_valid_psd_accepted():
    sd = FLife.SpectralData(input=psd_input())
    assert sd.moments[0] > 0


def test_ndarray_input_raises_value_error():
    with pytest.raises(ValueError, match='Unrecognized'):
        FLife.SpectralData(input=np.ones(10))


def test_f_not_strictly_increasing():
    f = F.copy()
    f[5] = f[4]
    with pytest.raises(ValueError, match='strictly increasing'):
        FLife.SpectralData(input=psd_input(f=f))


def test_negative_frequency():
    with pytest.raises(ValueError, match='non-negative'):
        FLife.SpectralData(input=psd_input(f=F - 1))


def test_negative_psd():
    psd = PSD.copy()
    psd[40] = -1.0
    with pytest.raises(ValueError, match='non-negative'):
        FLife.SpectralData(input=psd_input(psd=psd))


def test_zero_psd():
    with pytest.raises(ValueError, match='m0'):
        FLife.SpectralData(input=psd_input(psd=np.zeros_like(PSD)))


def test_amplitude_spectrum_negative_f():
    with pytest.raises(ValueError):
        FLife.SpectralData(input={'amplitude_spectrum': PSD, 'f': F - 1})


def _multiaxial(n_points=None, nf=40, df=3.0):
    f = np.arange(nf) * df
    f[0] = 1e-3
    psd = np.zeros((nf, 3, 3))
    psd[5:20] = np.eye(3) * 1.0
    if n_points:
        psd = np.stack([psd] * n_points)
    return {'PSD': psd, 'f': f}


def test_multiaxial_valid_accepted():
    FLife.SpectralData(input=_multiaxial(), T=1, fs=5000)
    FLife.SpectralData(input=_multiaxial(n_points=2), T=1, fs=5000)


def test_multiaxial_non_uniform_f():
    inp = _multiaxial()
    inp['f'][10:] += 0.5
    with pytest.raises(ValueError, match='uniform'):
        FLife.SpectralData(input=inp)


def test_multiaxial_negative_f():
    inp = _multiaxial()
    inp['f'] = inp['f'] - 10
    with pytest.raises(ValueError):
        FLife.SpectralData(input=inp)


def test_multiaxial_negative_psd():
    inp = _multiaxial()
    inp['PSD'][8, 0, 0] = -1.0
    with pytest.raises(ValueError, match='non-negative'):
        FLife.SpectralData(input=inp)


def test_multiaxial_f_length_mismatch():
    inp = _multiaxial()
    inp['f'] = inp['f'][:-1]
    with pytest.raises(ValueError):
        FLife.SpectralData(input=inp)


@pytest.mark.parametrize('n_points', [None, 2])
def test_multiaxial_fs_check_uses_frequency_axis(n_points):
    # content up to f[19] = 57 Hz in every component; fs = 100 < 2*57 must be rejected
    with pytest.raises(ValueError, match='fs'):
        FLife.SpectralData(input=_multiaxial(n_points=n_points), T=1, fs=100)


def test_multiaxial_fs_check_uses_highest_frequency_of_all_points():
    inp = _multiaxial(n_points=2)
    # second point has content only up to a lower frequency than the first
    inp['PSD'][1, 10:] = 0
    inp['PSD'][0, 20:] = 0
    inp['PSD'][0, 19] = np.eye(3)
    with pytest.raises(ValueError, match='fs'):
        FLife.SpectralData(input=inp, T=1, fs=100)


def test_multiaxial_shape_check_precedence():
    # 3D array with wrong trailing dims must be rejected (and/or precedence)
    with pytest.raises(Exception, match='size of'):
        FLife.SpectralData(input={'PSD': np.ones((40, 3, 4)), 'f': np.arange(40) * 3.0})
    with pytest.raises(Exception, match='size of'):
        FLife.SpectralData(input={'amplitude_spectrum': np.ones((40, 4)), 'f': np.arange(40) * 3.0})
