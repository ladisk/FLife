"""Multiaxial rainflow criterion against analytic references (synthetic PSDs only).

Pitoiset X, Preumont A. Spectral methods for multiaxial random fatigue analysis of
metallic structures. Int J Fatigue 2000;22:541-550.
"""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

F = np.arange(0, 201, 1.)
P1 = np.exp(-0.5 * ((F - 50) / 8)**2)
P2 = np.exp(-0.5 * ((F - 120) / 6)**2)
P1[0] = P2[0] = 0.


def random_state(seed):
    """Plane-stress PSD matrix (f, 3, 3) of two coherent sources in separate bands."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    return (np.einsum('i,j,f->fij', a[0], a[0].conj(), P1)
            + np.einsum('i,j,f->fij', a[1], a[1].conj(), P2))


def eq_psd(S, f=F, **kwargs):
    es = FLife.EquivalentStress(input={'PSD': S, 'f': f})
    es.multiaxial_rainflow(**kwargs)
    return es.eq_psd_multipoint[0] if es.multipoint else es.psd[:, 1]


@pytest.mark.parametrize('scale', [1e-8, 1e8, 1e12])
@pytest.mark.parametrize('seed', [0, 1])
def test_multiaxial_rainflow_scales_with_the_input(scale, seed):
    # A change of stress unit (e.g. MPa to Pa, scale 1e12 on the PSD) only scales the result.
    S = random_state(seed)
    np.testing.assert_allclose(eq_psd(scale * S), scale * eq_psd(S), rtol=1e-5)
