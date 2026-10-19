import sys, os
import numpy as np
import pytest
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife


def _psd_input():
    f = np.arange(0, 200, 0.5)
    psd = np.where((f > 20) & (f < 80), 1.0, 0.0)
    return {'PSD': psd, 'f': f}


def test_same_seed_same_signal():
    """rg passed to SpectralData must give an identical time history"""
    a = FLife.SpectralData(input=_psd_input(), T=5, fs=1000, rg=np.random.default_rng(1))
    b = FLife.SpectralData(input=_psd_input(), T=5, fs=1000, rg=np.random.default_rng(1))
    c = FLife.SpectralData(input=_psd_input(), T=5, fs=1000, rg=np.random.default_rng(2))
    assert np.array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)


def test_rainflow_requires_T_and_fs_together():
    sd = FLife.SpectralData(input=_psd_input())
    with pytest.raises(Exception, match='T and fs'):
        FLife.Rainflow(sd, T=5)
    with pytest.raises(Exception, match='T and fs'):
        FLife.Rainflow(sd, fs=1000)


def test_rainflow_nr_load_classes_used():
    sd = FLife.SpectralData(input=_psd_input(), T=20, fs=1000, rg=np.random.default_rng(3))
    rf = FLife.Rainflow(sd)
    coarse = rf.get_life(C=1e12, k=3, nr_load_classes=8)
    fine = rf.get_life(C=1e12, k=3, nr_load_classes=512)
    default = rf.get_life(C=1e12, k=3)
    assert coarse != fine
    assert fine == default
