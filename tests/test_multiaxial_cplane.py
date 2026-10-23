import numpy as np
import pytest

from FLife.multiaxial import cplane


F = np.arange(1., 101.)


def random_psd(seed):
    """Random Hermitian positive semidefinite 6x6 cross-PSD on F."""
    rng = np.random.default_rng(seed)
    S = np.zeros((len(F), 6, 6), dtype=complex)
    for _ in range(6):
        v = rng.normal(size=6) + 1j * rng.normal(size=6)
        g = np.exp(-0.5 * ((F - rng.uniform(10, 90)) / rng.uniform(3, 10))**2)
        S += g[:, None, None] * np.outer(v, v.conj())[None]
    return S


def plane_vector(method, dc):
    l1, m1, n1, l2, m2, n2, l3, m3, n3 = dc
    if method == 'maxnormal':
        return np.array([l1**2, m1**2, n1**2, 2*l1*m1, 2*l1*n1, 2*m1*n1])
    return np.array([l1**2 - l3**2, m1**2 - m3**2, n1**2 - n3**2,
                     2*(l1*m1 - l3*m3), 2*(l1*n1 - l3*n3), 2*(m1*n1 - m3*n3)])


def variance(method, dc, mu):
    a = plane_vector(method, dc)
    return np.einsum('i...,ij,j...->...', a, mu, a).real


def brute_force(method, mu, n=60):
    angles = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2*np.pi, 2*n), np.linspace(0, 2*np.pi, 2*n),
                         indexing='ij')
    return variance(method, cplane.compute_lmn_angles(*angles), mu).max()


@pytest.mark.parametrize('method', ['maxnormal', 'maxshear'])
@pytest.mark.parametrize('seed', [0, 1])
def test_critical_plane_does_not_depend_on_load_level(method, seed):
    # Nieslony and Macha (2007), Eq. (4.1): the critical plane follows from the direction cosines only,
    # so scaling the stress PSD by c**2 must not change the plane, and the variance scales by c**2
    S, df = random_psd(seed), 1.
    mu = np.sum(S, axis=0) * df
    reference = brute_force(method, mu)
    for c2 in (1e-8, 1., 1e8):
        dc = cplane.max_variance(S * c2, df, method)
        found = variance(method, dc, mu)
        assert found >= reference * (1 - 1e-9), f'c2={c2}: {found / reference}'
