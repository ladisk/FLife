import os

import numpy as np
import pytest

from FLife.multiaxial import cplane


F = np.arange(1., 101.)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'test_multiaxial_PSD_3D.npy')
K = 0.2

# Variance on the critical plane found by the multi-start SLSQP search over Euler angles
# (FLife main before the vectorised search). The new search must not find a worse plane.
REFERENCE = {
    'rand0': {'maxnormal': 304.6789861216058, 'maxshear': 793.7454130227138, 'maxnormalshear': 596.3837834868846},
    'rand1': {'maxnormal': 371.82330026604615, 'maxshear': 982.2235289237177, 'maxnormalshear': 744.2312777882761},
    'rand2': {'maxnormal': 633.8225324458685, 'maxshear': 1466.9830643109267, 'maxnormalshear': 1171.033060749436},
    'rand3': {'maxnormal': 616.205719689329, 'maxshear': 1539.103356994181, 'maxnormalshear': 1153.6304780621356},
    'plane0': {'maxnormal': 335.4086763750114, 'maxshear': 829.6387394303931, 'maxnormalshear': 648.0572126626755},
    'data331': {'maxnormal': 3829.864119276256, 'maxshear': 5521.174710655545, 'maxnormalshear': 5036.753973528309},
    'data334': {'maxnormal': 2951.4008854876715, 'maxshear': 4805.8715627865995, 'maxnormalshear': 4247.60625345909},
}


def random_psd(seed, n=6):
    """Random Hermitian positive semidefinite n x n cross-PSD on F."""
    rng = np.random.default_rng(seed)
    S = np.zeros((len(F), n, n), dtype=complex)
    for _ in range(n):
        v = rng.normal(size=n) + 1j * rng.normal(size=n)
        g = np.exp(-0.5 * ((F - rng.uniform(10, 90)) / rng.uniform(3, 10))**2)
        S += g[:, None, None] * np.outer(v, v.conj())[None]
    return S


def case_psd(name):
    if name.startswith('rand'):
        return random_psd(int(name[4:]))
    if name.startswith('plane'):
        return random_psd(100 + int(name[5:]), n=3)
    return np.load(DATA)[int(name[4:])]


def plane_vector(method, dc):
    """Voigt vector a of max_variance, so that a . sigma is the stress on the plane."""
    l1, m1, n1, l2, m2, n2, l3, m3, n3 = dc
    if method == 'maxnormal':
        return np.array([l1**2, m1**2, n1**2, 2*l1*m1, 2*l1*n1, 2*m1*n1])
    k = 0. if method == 'maxshear' else K
    return np.array([l1**2 - l3**2 + k*(l1 + l3)**2,
                     m1**2 - m3**2 + k*(m1 + m3)**2,
                     n1**2 - n3**2 + k*(n1 + n3)**2,
                     2*(l1*m1 - l3*m3 + k*(l1 + l3)*(m1 + m3)),
                     2*(l1*n1 - l3*n3 + k*(l1 + l3)*(n1 + n3)),
                     2*(m1*n1 - m3*n3 + k*(m1 + m3)*(n1 + n3))]) / (1 + k)


@pytest.mark.parametrize('method', ['maxnormal', 'maxshear', 'maxnormalshear'])
@pytest.mark.parametrize('name', list(REFERENCE))
def test_plane_variance_not_below_reference(name, method):
    S = case_psd(name)
    mu = np.real(np.sum(cplane._plane_stress_to_voigt_3d(S), axis=0))
    dc = cplane.max_variance(S, 1., method, K=K)
    a = plane_vector(method, dc)
    assert a @ mu @ a >= REFERENCE[name][method] * (1 - 1e-9)


@pytest.mark.parametrize('method', ['maxnormal', 'maxshear', 'maxnormalshear'])
def test_direction_cosines_are_right_handed_triad(method):
    dc = cplane.max_variance(random_psd(0), 1., method, K=K)
    assert len(dc) == 9
    Q = np.array(dc).reshape(3, 3)
    np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(Q), 1., atol=1e-12)


@pytest.mark.parametrize('method', ['maxnormal', 'maxshear'])
def test_uniaxial_stress(method):
    # sigma_x only: max normal variance on the x plane, max shear variance (2 tau)^2 = sigma_x^2 at 45 deg
    S = np.zeros((len(F), 6, 6))
    S[:, 0, 0] = 1.
    a = plane_vector(method, cplane.max_variance(S, 1., method))
    np.testing.assert_allclose(a @ np.sum(S, axis=0) @ a, len(F), rtol=1e-10)


def test_zero_psd():
    dc = cplane.max_variance(np.zeros((len(F), 6, 6)), 1., 'maxshear')
    assert np.all(np.isfinite(dc))


@pytest.mark.parametrize('method', ['maxnormal', 'maxshear', 'maxnormalshear'])
def test_local_search_refines_at_most_three_starts(monkeypatch, method):
    # the grid picks the start points, so SLSQP runs at most 3 times (16 before)
    calls = []
    minimize = cplane.opt.minimize

    def spy(*args, **kwargs):
        calls.append(1)
        return minimize(*args, **kwargs)

    monkeypatch.setattr(cplane.opt, 'minimize', spy)
    cplane.max_variance(random_psd(0), 1., method, K=K)
    assert 1 <= len(calls) <= 3
