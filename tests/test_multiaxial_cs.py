"""Carpinteri-Spagnoli criterion against analytic references (synthetic PSDs only).

Carpinteri A, Spagnoli A, Vantadori S. Reformulation in the frequency domain of a
critical plane-based multiaxial fatigue criterion. Int J Fatigue 2014;67:55-61.
"""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.multiaxial import cplane

F = np.arange(0, 201, 1.)

UNIAXIAL = np.diag([1., 0, 0])


def voigt(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]])


def psd_from_sources(tensors, psds, n=len(F)):
    """PSD matrix (f,6,6) of a sum of fully coherent sources: complex stress tensor times scalar PSD."""
    S = np.zeros((n, 6, 6), dtype=complex)
    for T, p in zip(tensors, psds):
        v = voigt(T)
        S += p[:, None, None] * np.outer(v, np.conj(v))[None]
    return S


def cs_psd(S, f=F, T=1., **kwargs):
    """Equivalent C-S PSD; T is given to EquivalentStress (the only way on older versions)."""
    es = FLife.EquivalentStress(input={"PSD": S, "f": f}, T=T, fs=10 * f.max())
    es.cs(**kwargs)
    return es.psd[:, 1]


def test_cs_frequency_vector():
    """The spectral moments, Eq. (15a/b), use the given frequency vector."""
    # np.arange(0, N*df, df) gives N+1 points here (202*0.1 = 20.200000000000003)
    f = np.arange(202) * 0.1
    S = psd_from_sources([UNIAXIAL], [np.ones(len(f))], n=len(f))
    np.testing.assert_allclose(cs_psd(S, f=f, T=10., s_af=1., tau_af=1.), 1.)
    assert cplane.csrandom(S, 0.1, 1., 1.).shape == (6, 6)
