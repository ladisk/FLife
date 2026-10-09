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
    if criterion == 'cs':
        kwargs.setdefault('T', 1.)
    if criterion == 'multiaxial_rainflow':
        kwargs.setdefault('k', 5.)
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
    'cs': {'s_af': S_AF, 'tau_af': TAU_AF},
    'thermoelastic': {},
    'Nieslony': {'s_af': S_AF, 'tau_af': TAU_AF},
    'Lemaitre': {'poisson_ratio': 0.3},
}
DELTA = 3 * np.pi / 8 * (1 - (TAU_AF / S_AF)**2)
UNIAXIAL_FACTOR = {
    'EVMS': 1., 'max_normal': 1., 'max_shear': 1., 'thermoelastic': 1., 'Nieslony': 1., 'Lemaitre': 1.,
    # the plane of max variance of the equivalent stress is not at 45 deg
    'max_normal_and_shear': ((np.sqrt(1 + K**2) + K) / (1 + K))**2,
    # normal and shear stress on the plane tilted by delta, Eq. 22 as printed
    'cs': np.cos(DELTA)**4 + S_AF / TAU_AF * np.sin(DELTA)**2 * np.cos(DELTA)**2,
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


@pytest.mark.parametrize('criterion', ['EVMS', 'Lemaitre', 'thermoelastic', 'multiaxial_rainflow',
                                       'EVMS_out_of_phase', 'max_normal_and_shear'])
def test_uniaxial_reduction_plane_stress(criterion):
    kwargs = {'poisson_ratio': 0.3} if criterion == 'Lemaitre' else {}
    if criterion == 'max_normal_and_shear':
        kwargs = {'s_af': S_AF, 'tau_af': TAU_AF}
    expected = UNIAXIAL_FACTOR.get(criterion, 1.)
    # multiaxial_rainflow (largest eigenvalue of the Voigt PSD) is not frame invariant
    for R in (IN_PLANE[:1] if criterion == 'multiaxial_rainflow' else IN_PLANE):
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
    # largest eigenvalue of Re(S) per frequency (the optimiser used to stop early)
    np.testing.assert_allclose(ratio(eq_psd(S, 'multiaxial_rainflow')), 1., rtol=1e-12)


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


@pytest.mark.parametrize('criterion', ['max_normal', 'max_shear', 'max_normal_and_shear', 'cs'])
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

@pytest.mark.parametrize('criterion', list(PSD_CRITERIA) + ['multiaxial_rainflow', 'EVMS_out_of_phase'])
def test_all_points_equal_single_points(criterion):
    plane = criterion in ('multiaxial_rainflow', 'EVMS_out_of_phase')
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


# --- Carpinteri-Spagnoli ----------------------------------------------------------------

def test_cs_delta_limited_as_in_authors_code():
    # tau_af/s_af < 1/sqrt(3): delta = pi/4 (C_S_Spettrale.m:382-393)
    for R in ROTATIONS[:3]:
        S = psd_from_sources([R @ UNIAXIAL @ R.T], [P])
        np.testing.assert_allclose(ratio(eq_psd(S, 'cs', s_af=300., tau_af=150.)), 0.25 + 2 * 0.25, rtol=1e-8)


def test_cs_number_of_cycles():
    """N1 = nu0*T with nu0 = sqrt(m2/m0) in Hz (Marques et al. 2020, Eq. 1 and 6)."""
    f = np.arange(0, 300, 0.5)
    px = np.exp(-0.5 * ((f - 20) / 1.)**2)
    py = np.exp(-0.5 * ((f - 200) / 1.)**2)
    px, py = px / px.sum(), 0.5 * py / py.sum()          # var_x = 1, var_y = 0.5 (per unit df)
    S = np.zeros((len(f), 6, 6))
    S[:, 0, 0], S[:, 1, 1] = px, py

    def davenport(var, nu0, T):
        x = np.sqrt(2 * np.log(nu0 * T))
        return np.sqrt(var) * (x + 0.5772 / x)

    for T in [1., 0.2]:
        # with delta = 0 the equivalent PSD is the normal stress PSD along 1-hat
        s_eq = eq_psd(S, 'cs', f=f, s_af=1., tau_af=1., T=T)
        x_wins = davenport(1., 20., T) > davenport(0.5, 200., T)
        np.testing.assert_allclose(s_eq, px if x_wins else py, atol=1e-10)
    assert davenport(1., 20., 1.) > davenport(0.5, 200., 1.)
    assert davenport(1., 20., 0.2) < davenport(0.5, 200., 0.2)


def test_cs_frequency_vector():
    # np.arange(0, N*df, df) used to give N+1 points for some (N, df)
    f = np.arange(24) * 0.1
    S = psd_from_sources([UNIAXIAL], [np.ones(len(F))])[:24]
    s_eq = eq_psd(S, 'cs', f=f, s_af=1., tau_af=1.)
    np.testing.assert_allclose(s_eq, 1.)
    assert cplane.csrandom(S, 0.1, 1., 1.).shape == (6, 6)


@pytest.mark.xfail(strict=True, reason='C-S: the sign of 3-hat (tilt direction) and 3-hat itself under '
                   'proportional loading are not defined by the source; the frame convention of '
                   'C_S_Spettrale.m is kept, so results depend on the reference frame. Maintainer decision.')
@pytest.mark.parametrize('case', ['general', 'pure shear'])
def test_cs_frame_independence(case):
    if case == 'general':
        tensors, psds = random_state(1), [P1, P2]
    else:
        tensors, psds = [SHEAR], [P]
    ref = eq_psd(psd_from_sources(tensors, psds), 'cs', s_af=S_AF, tau_af=TAU_AF)
    for R in ROTATIONS[:4]:
        S = psd_from_sources([R @ T @ R.T for T in tensors], psds)
        np.testing.assert_allclose(eq_psd(S, 'cs', s_af=S_AF, tau_af=TAU_AF), ref, rtol=1e-6)


# --- amplitude-spectrum criteria --------------------------------------------------------

def test_coin_liwi_uniaxial_phase():
    # Schmidt & Kraft (2023), Eq. 36: phase follows the first invariant
    a = np.zeros((5, 6), complex)
    a[:, 0] = np.exp(1j * 0.3)
    s_eq = criteria._coin_liwi(None, a, k_a=1.7, k_phi=0.9)
    np.testing.assert_allclose(np.angle(s_eq), 0.3, atol=1e-12)


def _liwi_states(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))


def test_liwi_amplitude_is_bonte_eq10():
    # Schmidt & Pandiya (2021), Eqs. 10-11: Bonte amplitude. This is COIN-LiWI with
    # k_a^2 = 3 and k_phi = 0.5 (Schmidt & Kraft 2023, Eq. 32), not k_phi = 1 (their Eq. 38).
    s = _liwi_states()
    sx, sy, t = s[:, 0], s[:, 1], s[:, 2]
    d = np.angle(sy) - np.angle(sx)
    H = abs(sx)**2 + abs(sy)**2 - abs(sx) * abs(sy) * np.cos(d) + 3 * abs(t)**2
    C = (abs(sx)**2 + abs(sy)**2 * np.exp(2j * d) - abs(sx) * abs(sy) * np.exp(1j * d)
         + 3 * abs(t)**2 * np.exp(2j * (np.angle(t) - np.angle(sx))))
    amp = np.abs(criteria._liwi(None, s))
    np.testing.assert_allclose(amp, np.sqrt(0.5 * H + 0.5 * np.abs(C)), rtol=1e-12)
    s6 = np.zeros((s.shape[0], 6), complex)
    s6[:, [0, 1, 3]] = s
    np.testing.assert_allclose(amp, np.abs(criteria._coin_liwi(None, s6, k_a=np.sqrt(3), k_phi=0.5)), rtol=1e-12)


@pytest.mark.parametrize('phase', [0., np.pi])
def test_liwi_static_sign(phase):
    # Schmidt & Pandiya (2021), Eqs. 32-33: in-phase states keep the sign of the stresses
    s = np.array([[2., 1., 0.5], [1., -0.3, 0.2], [0.4, 0., 1.]]) * np.exp(1j * phase)
    s_eq = criteria._liwi(None, s)
    np.testing.assert_allclose(s_eq.real, np.sign(np.cos(phase)) * np.abs(s_eq), atol=1e-12)


def _liwi_phase_reference(s):
    """Schmidt & Pandiya (2021), Eqs. 12 and 15-30, evaluated with the exact maximum times
    (Eqs. 17, 24) and branch rules (Eqs. 20, 26) instead of the approximation of Eqs. 27-28."""
    sx, sy, t = s[:, 0], s[:, 1], s[:, 2]
    A1, p1 = np.abs(sx), np.angle(sx)
    for A2, p2, th in [(np.abs(sy), np.angle(sy), 2 * np.pi / 3), (np.sqrt(3) * np.abs(t), np.angle(t), np.pi / 2)]:
        d = p2 - p1
        wt = -0.5 * np.angle(A1**2 + A2**2 * np.exp(2j * d) + 2 * A1 * A2 * np.exp(1j * d) * np.cos(th))
        a, b = A1 * np.cos(wt), A2 * np.cos(wt + d)                          # Eq. 16
        p_a = np.angle(A1 + A2 * np.exp(1j * (d - th)))                      # Eq. 18
        p_c = np.angle(a + b * np.exp(1j * th))                              # Eqs. 19, 25
        p_c = np.where((A2 > A1) & (p_c < th / 2), p_c + np.pi,
                       np.where((A2 <= A1) & (p_c > th / 2), th - p_c, p_c))  # Eqs. 20, 26
        A1 = np.abs(a + b * np.exp(1j * th))                                 # Eqs. 15, 22
        p1 = p_c + p_a + p1                                                  # Eqs. 21, 29
    p_bo = 0.5 * np.angle(sx**2 + sy**2 - sx * sy + 3 * t**2)                # Eq. 12
    closer = np.abs(np.angle(np.exp(1j * (p1 - p_bo)))) < np.abs(np.angle(np.exp(1j * (p1 - p_bo - np.pi))))
    return np.where(closer, p_bo, p_bo + np.pi)                              # Eq. 30


def test_liwi_phase_matches_article_equations():
    s = _liwi_states()
    s_eq = criteria._liwi(None, s)
    ref = np.exp(1j * _liwi_phase_reference(s))
    np.testing.assert_allclose(s_eq / np.abs(s_eq), ref, atol=1e-9)


def test_cs_requires_observation_time():
    S = psd_from_sources([UNIAXIAL], [P])
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    with pytest.raises(ValueError, match='observation time'):
        es.cs(s_af=1., tau_af=1.)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F}, T=2., fs=1000.)
    es.cs(s_af=1., tau_af=1.)
    ref = eq_psd(S, 'cs', s_af=1., tau_af=1., T=2.)
    np.testing.assert_allclose(es.psd[:, 1], ref)


def test_loop_over_points_deprecated():
    S = np.stack([psd_from_sources([UNIAXIAL], [P]), psd_from_sources([SHEAR], [P])])
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.EVMS()
    ref = es.eq_psd_multipoint[0]
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    with pytest.warns(DeprecationWarning):
        es.loop_over_points(criterion=criteria._EVMS)
    np.testing.assert_allclose(es.eq_psd_multipoint[0], ref)
    np.testing.assert_allclose(es.eq_psd_multipoint[1], F)


# --- multiaxial rainflow (Pitoiset and Preumont 2000) -----------------------------------

def _mr_state(seed):
    """Two coherent plane-stress sources in separate bands (3x3 PSD)."""
    return to_2d(psd_from_sources(random_state(seed, plane=True), [P1, P2]))


def _scan_directions(n=40000):
    rng = np.random.default_rng(1)
    c = rng.normal(size=(n, 3))
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def _moment(S, c, q):
    """Single-moment quantity lambda_q(c) = sum_f f^q c^T Re S c df (df = 1 here)."""
    psd = np.einsum('ni,fij,nj->nf', c, np.real(S), c)
    return psd @ (F**q)


@pytest.mark.parametrize('i', [0, 1, 2])
@pytest.mark.parametrize('k', [None, 3., 7.3])
def test_multiaxial_rainflow_uniaxial_reduction(i, k):
    S = np.zeros((len(F), 3, 3))
    S[:, i, i] = P
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        es.multiaxial_rainflow(k=k)
    np.testing.assert_allclose(es.psd[:, 1], P, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.abs(es.multiaxial_rainflow_direction), np.eye(3)[i], atol=1e-12)


def test_multiaxial_rainflow_two_bands_fixed_direction():
    # s_x at 50 Hz and uncorrelated shear at 120 Hz: a fixed c cannot have both
    S = np.zeros((len(F), 3, 3))
    S[:, 0, 0] = P1
    S[:, 2, 2] = 2 * P2
    eq = eq_psd(S, 'multiaxial_rainflow', k=5.)
    # damage-based choice at k = 5 takes the high-frequency shear band, not the envelope
    np.testing.assert_allclose(eq, 2 * P2, rtol=1e-12, atol=1e-15)
    envelope = np.linalg.eigvalsh(S)[:, -1]
    assert eq.sum() < envelope.sum()


@pytest.mark.parametrize('k', [3., 5., 10.])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_multiaxial_rainflow_equals_max_damage_scan(k, seed):
    S = _mr_state(seed)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.multiaxial_rainflow(k=k)
    c = es.multiaxial_rainflow_direction
    np.testing.assert_allclose(np.linalg.norm(c), 1.)
    q = 2. / k
    best = _moment(S, c[None], q)[0]
    scan = _moment(S, _scan_directions(), q)
    assert best >= scan.max() * (1 - 1e-12)
    assert best <= scan.max() * (1 + 1e-3)
    np.testing.assert_allclose(es.psd[:, 1], np.einsum('i,fij,j->f', c, np.real(S), c), rtol=1e-12)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_multiaxial_rainflow_without_k_is_max_variance(seed):
    S = _mr_state(seed)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    with pytest.warns(UserWarning, match='`k`'):
        es.multiaxial_rainflow()
    c = es.multiaxial_rainflow_direction
    best = _moment(S, c[None], 0.)[0]
    scan = _moment(S, _scan_directions(), 0.)
    assert best >= scan.max() * (1 - 1e-12)
    assert best <= scan.max() * (1 + 1e-3)


def test_multiaxial_rainflow_no_warning_with_k():
    S = _mr_state(0)
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        es.multiaxial_rainflow(k=4.)


def test_multiaxial_rainflow_sign_invariance_and_imaginary_part():
    S = _mr_state(3)
    r = np.real(S)
    eq = eq_psd(S, 'multiaxial_rainflow', k=4.)
    # c and -c give the same PSD
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.multiaxial_rainflow(k=4.)
    c = es.multiaxial_rainflow_direction
    np.testing.assert_allclose(np.einsum('i,fij,j->f', -c, r, -c), eq, rtol=1e-12)
    # only Re S matters
    A = np.random.default_rng(5).normal(size=(3, 3))
    S2 = S + 1j * (A - A.T)[None] * P[:, None, None]
    np.testing.assert_allclose(eq_psd(S2, 'multiaxial_rainflow', k=4.), eq, rtol=1e-10)


def test_multiaxial_rainflow_multipoint_direction():
    S = np.stack([_mr_state(s) for s in range(3)])
    es = FLife.EquivalentStress(input={'PSD': S, 'f': F})
    es.multiaxial_rainflow(k=4.)
    assert es.multiaxial_rainflow_direction.shape == (3, 3)
    single = FLife.EquivalentStress(input={'PSD': S[1], 'f': F})
    single.multiaxial_rainflow(k=4.)
    np.testing.assert_allclose(es.multiaxial_rainflow_direction[1], single.multiaxial_rainflow_direction)
