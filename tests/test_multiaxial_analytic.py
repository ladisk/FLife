"""Multiaxial criteria against analytic references (synthetic PSDs only)."""
import numpy as np
import pytest
import warnings
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.multiaxial import cplane, criteria

F = np.arange(0, 201, 1.)
P = np.exp(-0.5 * ((F - 50) / 8)**2) + 0.5 * np.exp(-0.5 * ((F - 120) / 6)**2)
P[0] = 0.
P1 = np.exp(-0.5 * ((F - 50) / 8)**2)
P2 = np.exp(-0.5 * ((F - 120) / 6)**2)
P1[0] = P2[0] = 0.

S_AF, TAU_AF = 300., 180.          # s_af != tau_af, so K != 1 and delta != 0
K = 2 * TAU_AF / S_AF - 1

UNIAXIAL = np.diag([1., 0, 0])
SHEAR = np.array([[0., 1, 0], [1, 0, 0], [0, 0, 0]])


def rotation(a, b, c):
    """Rotation matrix from ZYZ angles."""
    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    def ry(t):
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    return rz(a) @ ry(b) @ rz(c)


ROTATIONS = [rotation(*a) for a in np.random.default_rng(0).uniform(0, 2*np.pi, size=(10, 3))]
IN_PLANE = [rotation(a, 0, 0) for a in (0., 0.3, 0.8, 2.0, 4.0)]


def voigt(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]])


def psd_from_sources(tensors, psds):
    """PSD matrix (f,6,6) of a sum of fully coherent sources: complex stress tensor times scalar PSD."""
    S = np.zeros((len(F), 6, 6), dtype=complex)
    for T, p in zip(tensors, psds):
        v = voigt(T)
        S += p[:, None, None] * np.outer(v, np.conj(v))[None]
    return S


def to_2d(S):
    idx = [0, 1, 3]
    return S[..., idx, :][..., :, idx]


def random_state(seed, plane=False):
    rng = np.random.default_rng(seed)
    tensors = []
    for _ in range(2):
        A = np.zeros((3, 3), complex)
        n = 2 if plane else 3
        B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        A[:n, :n] = (B + B.T) / 2
        tensors.append(A)
    return tensors


def eq_psd(S, criterion, f=F, **kwargs):
    es = FLife.EquivalentStress(input={'PSD': S, 'f': f})
    getattr(es, criterion)(**kwargs)
    return es.eq_psd_multipoint[0] if es.multipoint else es.psd[:, 1]


def ratio(s_eq):
    """Equivalent PSD over the input PSD P (where P is not negligible)."""
    m = P > 1e-6 * P.max()
    return np.real(s_eq[m]) / P[m]


PSD_CRITERIA = {
    'EVMS': {},
    'max_normal': {},
    'max_shear': {},
    'max_normal_and_shear': {'s_af': S_AF, 'tau_af': TAU_AF},
    'thermoelastic': {},
    'Nieslony': {'s_af': S_AF, 'tau_af': TAU_AF},
    'Lemaitre': {'poisson_ratio': 0.3},
}
UNIAXIAL_FACTOR = {
    'EVMS': 1., 'max_normal': 1., 'max_shear': 1., 'thermoelastic': 1., 'Nieslony': 1., 'Lemaitre': 1.,
    # the plane of max variance of the equivalent stress is not at 45 deg
    'max_normal_and_shear': ((np.sqrt(1 + K**2) + K) / (1 + K))**2,
}
SHEAR_FACTOR = {
    'EVMS': 3., 'max_normal': 1., 'max_shear': 4., 'thermoelastic': 0., 'Lemaitre': 2 * 1.3,
    'max_normal_and_shear': 4 * (1 + K**2) / (1 + K)**2,
    'Nieslony': (S_AF / TAU_AF)**2,
}
# Nieslony is left out: its phase reference is the component with the largest
# amplitude, which depends on the frame (differences of about 0.1 % here).
INVARIANT = ['EVMS', 'max_normal', 'max_shear', 'max_normal_and_shear', 'thermoelastic', 'Lemaitre']


# --- uniaxial reduction and pure shear --------------------------------------------------

@pytest.mark.parametrize('criterion', PSD_CRITERIA)
def test_uniaxial_reduction_any_orientation(criterion):
    for R in ROTATIONS[:4] + [np.eye(3)]:
        S = psd_from_sources([R @ UNIAXIAL @ R.T], [P])
        np.testing.assert_allclose(ratio(eq_psd(S, criterion, **PSD_CRITERIA[criterion])),
                                   UNIAXIAL_FACTOR[criterion], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize('criterion', ['EVMS', 'Lemaitre', 'thermoelastic',
                                       'EVMS_out_of_phase', 'max_normal_and_shear'])
def test_uniaxial_reduction_plane_stress(criterion):
    kwargs = {'poisson_ratio': 0.3} if criterion == 'Lemaitre' else {}
    if criterion == 'max_normal_and_shear':
        kwargs = {'s_af': S_AF, 'tau_af': TAU_AF}
    expected = UNIAXIAL_FACTOR.get(criterion, 1.)
    for R in IN_PLANE:
        S = to_2d(psd_from_sources([R @ UNIAXIAL @ R.T], [P]))
        np.testing.assert_allclose(ratio(eq_psd(S, criterion, **kwargs)), expected, rtol=1e-8)


@pytest.mark.parametrize('criterion', SHEAR_FACTOR)
def test_pure_shear_factors(criterion):
    for R in ROTATIONS[:3] + [np.eye(3)]:
        S = psd_from_sources([R @ SHEAR @ R.T], [P])
        np.testing.assert_allclose(ratio(eq_psd(S, criterion, **PSD_CRITERIA[criterion])),
                                   SHEAR_FACTOR[criterion], rtol=1e-8, atol=1e-10)


def test_pure_shear_plane_stress():
    S = to_2d(psd_from_sources([SHEAR], [P]))
    np.testing.assert_allclose(ratio(eq_psd(S, 'EVMS_out_of_phase')), 3., rtol=1e-12)


# --- invariance -------------------------------------------------------------------------

@pytest.mark.parametrize('criterion', INVARIANT)
def test_rotation_invariance_general_state(criterion):
    tensors = random_state(1)
    ref = eq_psd(psd_from_sources(tensors, [P1, P2]), criterion, **PSD_CRITERIA[criterion])
    for R in ROTATIONS:
        S = psd_from_sources([R @ T @ R.T for T in tensors], [P1, P2])
        np.testing.assert_allclose(eq_psd(S, criterion, **PSD_CRITERIA[criterion]), ref,
                                   rtol=1e-7, atol=1e-9 * ref.max())


@pytest.mark.parametrize('criterion', INVARIANT + ['EVMS_out_of_phase'])
def test_rotation_invariance_plane_stress(criterion):
    tensors = random_state(2, plane=True)
    psds = [P1, P2]
    if criterion == 'EVMS_out_of_phase':
        # Bonte assumes fully coherent components (sqrt(Sxx Syy)): one source only
        tensors, psds = tensors[:1], [P]
    kwargs = PSD_CRITERIA.get(criterion, {})
    ref = eq_psd(to_2d(psd_from_sources(tensors, psds)), criterion, **kwargs)
    for R in IN_PLANE:
        S = to_2d(psd_from_sources([R @ T @ R.T for T in tensors], psds))
        np.testing.assert_allclose(eq_psd(S, criterion, **kwargs), ref, rtol=1e-7, atol=1e-9 * ref.max())


@pytest.mark.parametrize('criterion', ['max_normal', 'max_shear', 'max_normal_and_shear'])
def test_scale_invariance(criterion):
    S = psd_from_sources(random_state(3), [P1, P2])
    ref = eq_psd(S, criterion, **PSD_CRITERIA[criterion])
    for scale in [1e-8, 1e-4, 1e4, 1e8, 1e12]:
        # the optimum is flat, so the plane itself is found to about 1e-8 rad
        np.testing.assert_allclose(eq_psd(S * scale, criterion, **PSD_CRITERIA[criterion]) / scale, ref,
                                   rtol=1e-6, atol=1e-7 * ref.max())


def test_critical_plane_is_global_maximum():
    """The plane search matches a dense brute-force scan of the Euler angles."""
    S = psd_from_sources(random_state(4), [P1, P2])
    C = np.real(S.sum(0))
    th, ph, ps = np.meshgrid(np.linspace(0, np.pi, 31), np.linspace(0, 2*np.pi, 61),
                             np.linspace(0, 2*np.pi, 61), indexing='ij')
    L = np.array(cplane.compute_lmn_angles(th.ravel(), ph.ravel(), ps.ravel())).reshape(3, 3, -1)
    L = np.moveaxis(L, -1, 0)
    for method, k in [('maxnormal', None), ('maxshear', 0.), ('maxnormalshear', K)]:
        a = criteria._plane_vector(L, k)
        brute = np.einsum('gi,ij,gj->g', a, C, a).max()
        found = np.array(cplane.max_variance(S, 1., method, K=K)).reshape(3, 3)
        np.testing.assert_allclose(found @ found.T, np.eye(3), atol=1e-12)
        a = criteria._plane_vector(found, k)
        assert a @ C @ a >= brute * (1 - 1e-12)


# --- phase-sensitive criteria -----------------------------------------------------------

def test_90_degree_out_of_phase():
    # sigma_x and sigma_y with equal PSD, 90 deg out of phase
    S = to_2d(psd_from_sources([np.diag([1., 1j, 0])], [P]))
    np.testing.assert_allclose(ratio(eq_psd(S, 'EVMS')), 2., rtol=1e-12)
    np.testing.assert_allclose(ratio(eq_psd(S, 'EVMS_out_of_phase')), 1.5, rtol=1e-12)


def test_EVMS_out_of_phase_rejects_3d():
    S = psd_from_sources([SHEAR], [P])
    with pytest.raises(ValueError):
        eq_psd(S, 'EVMS_out_of_phase')


def test_closed_form_criteria_return_real_psd():
    S = psd_from_sources(random_state(5), [P1, P2])
    for criterion in ['EVMS', 'thermoelastic', 'Lemaitre']:
        kwargs = PSD_CRITERIA[criterion]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            single = eq_psd(S, criterion, **kwargs)
            multi = eq_psd(np.stack([S, 2 * S]), criterion, **kwargs)
        assert not np.iscomplexobj(single) and not np.iscomplexobj(multi)
        np.testing.assert_allclose(multi[1], 2 * single, rtol=1e-12)


# --- multipoint -------------------------------------------------------------------------

@pytest.mark.parametrize('criterion', list(PSD_CRITERIA) + ['EVMS_out_of_phase'])
def test_all_points_equal_single_points(criterion):
    plane = criterion == 'EVMS_out_of_phase'
    S = np.stack([psd_from_sources(random_state(s, plane=plane), [P1, P2]) for s in range(3)])
    if plane:
        S = to_2d(S)
    kwargs = PSD_CRITERIA.get(criterion, {})
    multi = eq_psd(S, criterion, **kwargs)
    for i in range(len(S)):
        np.testing.assert_allclose(multi[i], eq_psd(S[i], criterion, **kwargs), rtol=1e-9,
                                   atol=1e-12 * multi.max())


def test_multipoint_respects_search_method(monkeypatch):
    calls = []
    original = cplane.max_variance_planes

    def spy(*args, **kwargs):
        calls.append(kwargs.get('search_method'))
        return original(*args, **kwargs)

    monkeypatch.setattr(cplane, 'max_variance_planes', spy)
    S = np.stack([psd_from_sources(random_state(s), [P1, P2]) for s in range(2)])
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.max_normal(search_method='global')
    assert calls == ['global']


