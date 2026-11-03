import numpy as np
import scipy.optimize as opt

def _plane_stress_to_voigt_3d(multiaxial_psd):
    """Promote plane-stress PSD [sx, sy, txy] to 3D Voigt [sx, sy, sz, txy, txz, tyz].

    The critical-plane implementation operates on 6x6 Voigt stress transforms.
    For documented 2D PSD input, treat the out-of-plane terms as zero and embed
    the in-plane terms at Voigt indices [0, 1, 3].
    """
    if multiaxial_psd.shape[-2:] == (6, 6):
        return multiaxial_psd
    if multiaxial_psd.shape[-2:] != (3, 3):
        raise ValueError('Input Error. PSD matrix should be the size of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state')

    psd_3d = np.zeros(multiaxial_psd.shape[:-2] + (6, 6), dtype=multiaxial_psd.dtype)
    plane_stress_indices = [0, 1, 3]

    for row_2d, row_3d in enumerate(plane_stress_indices):
        for col_2d, col_3d in enumerate(plane_stress_indices):
            psd_3d[..., row_3d, col_3d] = multiaxial_psd[..., row_2d, col_2d]

    return psd_3d

def compute_lmn_angles(theta, phi, psi):
    """Direction cosines of a proper ZXZ-Euler rotation (theta, phi, psi).

    Returns the three rows (l_i, m_i, n_i) of an orthonormal direction-cosine
    matrix, so that (l1,m1,n1), (l2,m2,n2) and (l3,m3,n3) form a right-handed
    orthonormal triad for any input angles.
    """
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    l1 = cps * cph - cth * sph * sps
    m1 = cps * sph + cth * cph * sps
    n1 = sps * sth

    l2 = -sps * cph - cth * sph * cps
    m2 = -sps * sph + cth * cph * cps
    n2 = cps * sth

    l3 = sth * sph
    m3 = -sth * cph
    n3 = cth

    return l1, m1, n1, l2, m2, n2, l3, m3, n3


def _maximize(crit, x0, bounds, search_method):
    """Maximize -crit over the plane orientation with differential evolution ('global').

    The 'local' search of max_variance is _plane_search. Results are reproducible.
    """
    if search_method == 'global':
        return opt.differential_evolution(crit, bounds=bounds, seed=0)
    else:
        raise ValueError('Search method')


# Fast plane search for max_variance(search_method='local')
#
# With the traction t = sigma n on the plane with normal n, its covariance
# M(n) = Cov(t, t) (3x3) gives all three objectives. With e1 = (n + s) / sqrt(2)
# and e3 = (n - s) / sqrt(2), s _|_ n the shear direction, the variances of
# a . sigma in max_variance are
#   maxnormal:       Var(n^T sigma n)                         = n^T M n
#   maxshear:        Var(2 s^T sigma n)                       = 4 s^T M s
#   maxnormalshear:  Var(2 (s + K n)^T sigma n) / (1 + K)^2   = 4 q^T M q / (1 + K)^2, q = s + K n
# For 'maxshear' the best s is the top eigenvector of M projected on the plane
# (closed form), so the search is over n only. All normals of a grid on the half
# sphere are evaluated at once, then the best few grid points are refined locally.

_VOIGT_INDEX = np.array([[0, 3, 4],
                         [3, 1, 5],
                         [4, 5, 2]])


def _traction_operator(mu):
    """Matrix (9, 9) that maps n_i n_k (flattened) to the traction covariance M_jl (flattened)."""
    full = mu[_VOIGT_INDEX[:, :, None, None], _VOIGT_INDEX[None, None, :, :]]  # Cov(sigma_ji, sigma_lk)
    return full.transpose(0, 2, 1, 3).reshape(9, 9)


def _traction_covariance(op, n):
    """Traction covariance M (..., 3, 3) for normals n (..., 3)."""
    nn = (n[..., :, None] * n[..., None, :]).reshape(n.shape[:-1] + (9,))
    return (nn @ op.T).reshape(n.shape[:-1] + (3, 3))


def _hemisphere_grid(n_points=1000):
    """Nearly uniform unit normals on the half sphere z > 0 (Fibonacci lattice, spacing about 4.5 deg)."""
    i = np.arange(n_points) + 0.5
    z = 1 - i / n_points
    r = np.sqrt(1 - z**2)
    phi = np.pi * (3 - np.sqrt(5)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def _basis(n):
    """Unit vectors u, w so that (n, u, w) is a right-handed orthonormal triad."""
    use_x = np.abs(n[..., :1]) < 0.9
    a = np.where(use_x, [1., 0., 0.], [0., 1., 0.])
    u = a - np.sum(a * n, axis=-1, keepdims=True) * n
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return u, np.cross(n, u)


def _plane_objective(op, n, method, K, s=None):
    """Variance of a . sigma (see above) for normals n (..., 3) and shear directions s ('maxnormalshear')."""
    M = _traction_covariance(op, n)
    if method == 'maxnormalshear':
        q = s + K * n
        return 4 * np.einsum('...i,...ij,...j->...', q, M, q) / (1 + K)**2
    v = np.einsum('...ij,...j->...i', M, n)
    p = np.sum(n * v, axis=-1)
    if method == 'maxnormal':
        return p
    # top eigenvalue of P M P, P = I - n n^T: its other eigenvalues are 0 and the second shear one
    tr = np.trace(M, axis1=-2, axis2=-1) - p
    tr2 = np.sum(M * M, axis=(-2, -1)) - 2 * np.sum(v * v, axis=-1) + p**2
    return 2 * (tr + np.sqrt(np.maximum(2 * tr2 - tr**2, 0.)))


def _plane_search(mu, method, K, n_starts=3, n_alpha=36):
    """Direction cosines (l1, m1, n1, l2, m2, n2, l3, m3, n3) of the max-variance plane.

    mu is the real stress covariance (6, 6). The search starts from a grid on the
    half sphere (n and -n give the same plane) and refines the best grid point of
    up to n_starts separate peaks with SLSQP in a local chart around each point.
    """
    op = _traction_operator(mu)
    grid = _hemisphere_grid()
    u, w = _basis(grid)
    if method == 'maxnormalshear':
        # q^T M q for q = cos(alpha) u + sin(alpha) w + K n, from the entries of M in the basis (u, w, n)
        alpha = np.linspace(0, 2*np.pi, n_alpha, endpoint=False)
        M = _traction_covariance(op, grid)
        B = np.stack([u, w, grid], axis=-1)
        Mb = np.einsum('gji,gjk,gkl->gil', B, M, B)
        c, sn = np.cos(alpha), np.sin(alpha)
        Muu, Muw, Mww, Mun, Mwn, Mnn = (Mb[:, i, j, None] for i, j in [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)])
        values = 4 * (c**2 * Muu + 2 * c * sn * Muw + sn**2 * Mww
                      + 2 * K * (c * Mun + sn * Mwn) + K**2 * Mnn) / (1 + K)**2
        best_alpha = alpha[np.argmax(values, axis=1)]
        values = values.max(axis=1)
    else:
        values = _plane_objective(op, grid, method, K)

    # start points: best grid point of each separate peak (more than 15 deg apart). For 'maxshear'
    # the plane with normal s and shear direction n is the same peak, so skip that one too.
    starts, covered = [], []
    for i in np.argsort(-values, kind='stable'):
        if len(starts) == n_starts or values[i] < 0.95 * values[starts[0] if starts else i]:
            break
        if all(abs(grid[i] @ d) < np.cos(np.radians(15)) for d in covered):
            starts.append(i)
            covered.append(grid[i])
            if method == 'maxshear':
                x1, x2 = _max_shear_direction(mu, grid[i], u[i], w[i])
                covered.append(x1 * u[i] + x2 * w[i])

    def frame(x, i):
        """Normal n and, for 'maxnormalshear', shear direction s in a local chart around grid point i."""
        n = grid[i] + x[0] * u[i] + x[1] * w[i]
        n = n / np.linalg.norm(n)
        if method != 'maxnormalshear':
            return n, None
        a = u[i] - (u[i] @ n) * n
        a = a / np.linalg.norm(a)
        b = np.cross(n, a)
        return n, np.cos(best_alpha[i] + x[2]) * a + np.sin(best_alpha[i] + x[2]) * b

    n_vars = 3 if method == 'maxnormalshear' else 2
    best_value, best_frame = -np.inf, None
    for i in starts:
        def crit(x):
            n, s = frame(x, i)
            return -_plane_objective(op, n, method, K, s)

        res = opt.minimize(crit, np.zeros(n_vars), method='SLSQP', options={'ftol': 1e-14})
        x = res.x if res.fun < -values[i] else np.zeros(n_vars)
        value = -crit(x)
        if value > best_value:
            best_value, best_frame = value, frame(x, i)

    n, s = best_frame
    if method == 'maxnormal':
        e1 = n
        e2, e3 = _basis(n)
    else:
        if method == 'maxshear':
            a, b = _basis(n)
            x1, x2 = _max_shear_direction(mu, n, a, b)
            s = x1 * a + x2 * b
        e1, e3 = (n + s) / np.sqrt(2), (n - s) / np.sqrt(2)
        e2 = np.cross(e3, e1)
    return tuple(np.concatenate([e1, e2, e3]))


def max_variance(multiaxial_psd, df, method, K=None, search_method='local'):
    '''Determine critical plane
    based on max variance.

    returns: l1, m1, n1, l2, m2, n2, l3, m3, n3

    search_method: 'local' (default) evaluates a grid of plane normals and
    refines the best ones locally; 'global' uses differential evolution.

        -Lagoda, Macha and Nieslony; Fatigue Life Calculation
        by means of the cycle counting and spectral methods
        under multiaxial random loading.

        -Bȩdkowski, W., and E. Macha; Fatigue fracture plane
        under multiaxial random loadings–prediction by variance
        of equivalent stress based on the maximum shear and normal stresses.
    '''
    multiaxial_psd = _plane_stress_to_voigt_3d(multiaxial_psd)
    mu = np.sum(multiaxial_psd, axis=0) * df
    # The plane depends only on the direction cosines, not on the load level, but SLSQP
    # uses absolute tolerances. Scaling mu to unit trace keeps the objective O(1), as in csrandom.
    scale = np.real(np.trace(mu))
    if scale > 0:
        mu = mu / scale
    bounds = [(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)]

    if 'maxnormal' == method:

        def crit(v):
            """Criterion for max-normal."""
            theta, phi, psi = v
            l1, m1, n1, l2, m2, n2, l3, m3, n3 = compute_lmn_angles(theta, phi, psi)
            a = np.asarray([l1**2, m1**2, n1**2,
                            2*l1*m1, 2*l1*n1, 2*m1*n1])


            return -np.abs(np.einsum('i,ij,j', a, mu, a).real)

        x0 = [np.pi, np.pi, np.pi]


    elif 'maxshear' == method:

        def crit(v):
            """Criterion for max-shear."""
            theta, phi, psi = v
            l1, m1, n1, l2, m2, n2, l3, m3, n3 = compute_lmn_angles(theta, phi, psi)
            a = np.asarray([l1**2 - l3**2, m1**2 - m3**2, n1**2 - n3**2,
                            2 * (l1 * m1 - l3 * m3), 2 * (l1 * n1 - l3 * n3), 2 * (m1 * n1 - m3 * n3)])

            return -np.abs(np.einsum('i,ij,j', a, mu, a).real)

        x0 = [np.pi, np.pi, np.pi]

    elif 'maxnormalshear' == method:
        if K is None:
            raise ValueError

        def crit(v):
            """Criterion for max-normal-and-shear."""
            theta, phi, psi = v
            l1, m1, n1, l2, m2, n2, l3, m3, n3 = compute_lmn_angles(theta, phi, psi)
            a = np.asarray([l1**2 - l3**2 + K * (l1 + l3)**2,
                            m1**2 - m3**2 + K * (m1 + m3)**2,
                            n1**2 - n3**2 + K * (n1 + n3)**2,
                            2 * (l1 * m1 - l3 * m3 + K * (l1 + l3) * (m1 + m3)),
                            2 * (l1 * n1 - l3 * n3 + K * (l1 + l3) * (n1 + n3)),
                            2 * (m1 * n1 - m3 * n3 + K * (m1 + m3) * (n1 + n3))]) / (1 + K)

            return -np.abs(np.einsum('i,ij,j', a, mu, a).real)

        x0 = [np.pi/3, np.pi/3, np.pi/3]

    else:
        raise ValueError('Critical plane method')

    if search_method == 'local':
        return _plane_search(np.real(mu), method, K)
    res = _maximize(crit, x0, bounds, search_method)
    direction_cosines = compute_lmn_angles(*res['x'])

    return direction_cosines


# Carpinteri-Spagnoli
#
# 6x6 Voigt stress transformation matrices derived from first principles.
# Stress vector ordering: {sigma_x, sigma_y, sigma_z, tau_xy, tau_xz, tau_yz}
#
# Direction cosine matrix convention (passive/coordinate transform):
#   Q_z(a) = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
#   Q_x(a) = [[1, 0, 0], [0, c, s], [0, -s, c]]
# where c = cos(a), s = sin(a).
#
# The C-S frame is the ZXZ rotation Tz(chi) Tx(delta) Tz(psi) Tx(theta) Tz(phi):
#   phi, theta -- direction of the 1-hat axis (w)
#   psi        -- rotation about w (shear direction v)
#   delta      -- material off-angle about the u axis (Eq. 19)
#   chi        -- rotation about the tilted w (shear direction on the critical plane)
# csrandom finds the same frame by working on the axis vectors directly, which
# avoids the Euler-angle singularities. voigt_transform builds the 6x6 matrix.

def Tz(c, s):
    '''6x6 stress transformation for rotation about Z-axis.
    Leaves sigma_z (index 2) invariant.

    Parameters
    ----------
    c : float
        cos(angle)
    s : float
        sin(angle)
    '''
    return np.array([
        [ c**2,  s**2, 0,  2*c*s,      0,  0],
        [ s**2,  c**2, 0, -2*c*s,      0,  0],
        [ 0,     0,    1,  0,           0,  0],
        [-c*s,   c*s,  0,  c**2-s**2,  0,  0],
        [ 0,     0,    0,  0,           c,  s],
        [ 0,     0,    0,  0,          -s,  c]])

def Tx(c, s):
    '''6x6 stress transformation for rotation about X-axis.
    Leaves sigma_x (index 0) invariant.

    Parameters
    ----------
    c : float
        cos(angle)
    s : float
        sin(angle)
    '''
    return np.array([
        [1,  0,     0,    0,    0,     0        ],
        [0,  c**2,  s**2, 0,    0,     2*c*s    ],
        [0,  s**2,  c**2, 0,    0,    -2*c*s    ],
        [0,  0,     0,    c,    s,     0        ],
        [0,  0,     0,   -s,    c,     0        ],
        [0, -c*s,   c*s,  0,    0,     c**2-s**2]])


def spectral_moment(f, multiaxial_psd, n):
    """Calculate n-th spectral moment of a PSD.

    f: frequency array
    s: PSD array
    n: order of spectral moment
    
    Returns
    -------
    Scalar.
    """
    df = f[1] - f[0]
    
    #return np.tensordot(f**n, s, ([0], [0])) * df
    return np.einsum('i,i...->...', f**n, multiaxial_psd) * df


_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def voigt_transform(Q):
    '''6x6 Voigt stress transformation for direction cosines Q (3, 3).

    sigma' = Q sigma Q^T in Voigt form {sigma_x, sigma_y, sigma_z, tau_xy, tau_xz, tau_yz}.
    Tz and Tx are special cases.
    '''
    Q = np.asarray(Q)
    T = np.empty(Q.shape[:-2] + (6, 6))
    for I, (i, j) in enumerate(_VOIGT_PAIRS):
        for J, (k, l) in enumerate(_VOIGT_PAIRS):
            T[..., I, J] = Q[..., i, k] * Q[..., j, l]
            if k != l:
                T[..., I, J] += Q[..., i, l] * Q[..., j, k]
    return T


def _normal_vector(n):
    """Voigt coefficients a with a . sigma = n^T sigma n."""
    l, m, k = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([l*l, m*m, k*k, 2*l*m, 2*l*k, 2*m*k], axis=-1)


def _shear_vector(s, n):
    """Voigt coefficients a with a . sigma = s^T sigma n."""
    return np.stack([s[..., 0]*n[..., 0], s[..., 1]*n[..., 1], s[..., 2]*n[..., 2],
                     s[..., 0]*n[..., 1] + s[..., 1]*n[..., 0],
                     s[..., 0]*n[..., 2] + s[..., 2]*n[..., 0],
                     s[..., 1]*n[..., 2] + s[..., 2]*n[..., 1]], axis=-1)


def _davenport_extreme(k0, k2, T):
    """Davenport expected maximum sqrt(k0) * (sqrt(2 ln N1) + 0.5772 / sqrt(2 ln N1)), Eq. 17.

    N1 = nu0 * T (Eq. 16), with nu0 = sqrt(lambda2 / lambda0) / (2 pi) (Eq. 5) for
    moments in rad/s, which is sqrt(k2 / k0) for moments k in Hz. The value is 0
    where N1 <= 1 (the formula is undefined there).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        N1 = np.sqrt(np.where(k0 > 0, k2 / k0, 0.)) * T
        x = np.sqrt(2 * np.log(np.where(N1 > 1, N1, np.e)))
        value = np.sqrt(np.maximum(k0, 0.)) * (x + 0.5772 / x)
    return np.where((N1 > 1) & (k2 > 0), value, 0.)


def _matlab_sign(x, y, z):
    """+1 where the vector comes first in the authors' grid order, else -1.

    In the authors' MATLAB code the first strict maximum wins, so of two equivalent
    directions the one with the smaller angle (phi or theta) is kept.
    """
    first = np.where(y != 0, y > 0, np.where(x != 0, x > 0, z >= 0))
    return np.where(first, 1., -1.)


def _max_shear_direction(cov, n, a, b):
    """Coefficients (x1, x2) of the unit vector x1 a + x2 b (a, b _|_ n) that maximises
    the variance of the shear stress x^T sigma n (closed form: top eigenvector)."""
    ga, gb = _shear_vector(a, n), _shear_vector(b, n)
    m11, m12, m22 = ga @ cov @ ga, ga @ cov @ gb, gb @ cov @ gb
    d = 0.5 * (m11 - m22)
    rad = np.hypot(d, m12)
    x1, x2 = (d + rad, m12) if d >= 0 else (m12, rad - d)
    norm = np.hypot(x1, x2)
    if norm == 0:
        return 1., 0.
    return x1 / norm, x2 / norm


def csrandom(multiaxial_psd, df, s_af, tau_af, f=None, T=1.):
    '''Determine critical plane with the C-S random criterion.

    Returns the 6x6 Voigt transformation CC to the critical-plane frame:
    after transformation, sigma_w = S[2,2] and tau_vw = S[5,5].

    :param f: frequency vector in Hz for the spectral moments (Eq. 15a/b);
        defaults to arange(len) * df
    :param T: time span in s for the number of cycles N1 = nu0 * T (Eq. 16)

    The steps follow Sec. 3 and 4 of the article and the authors' MATLAB code
    (Carpinteri, Spagnoli, Vantadori, C_S_Spettrale.m):
        1. 1-hat maximises the Davenport expected maximum of the normal stress,
           Eq. 17, over 0 <= phi <= 2 pi, 0 <= theta <= pi. The authors scan a
           10 deg grid; here a 5 deg grid plus a local refinement is used.
        2. 3-hat _|_ 1-hat maximises the variance of the shear stress on the
           plane 1-hat (closed form instead of the 10 deg scan).
        3. w = cos(delta) 1-hat + sin(delta) 3-hat (Eq. 19 for delta).
        4. v _|_ w maximises the variance of tau_vw, Eq. 20 (closed form).
    Where two directions are equivalent (n and -n), the one the authors' scan
    meets first is kept, so the frame convention matches the authors' code.

    See also
    --------
    Carpinteri A, Spagnoli A and Vantadori S, Reformulation in the frequency
    domain of a critical plane-based multiaxial fatigue criterion,
    Int J Fat, 2014
    '''
    multiaxial_psd = _plane_stress_to_voigt_3d(np.asarray(multiaxial_psd))
    if f is None:
        f = np.arange(multiaxial_psd.shape[0]) * df
    f = np.asarray(f, dtype=float)
    l0 = np.real(spectral_moment(f, multiaxial_psd, 0))
    m2 = np.real(spectral_moment(f, multiaxial_psd, 2))

    # Normalise the moments before the plane search. The critical plane depends
    # only on the relative stress distribution, but SLSQP uses absolute
    # tolerances, so an objective that scales with the PSD magnitude converges to
    # a slightly different plane at different load levels. Scaling l0 and m2 by a
    # common factor keeps the objective O(1) and leaves both the plane and the
    # moment ratio in N1 unchanged. The equivalent stress is computed later in
    # _cs from the unscaled stress.
    scale = np.trace(l0)
    if scale > 0:
        l0 = l0 / scale
        m2 = m2 / scale

    def axis(phi, gamma):
        return np.stack([np.sin(gamma) * np.cos(phi), np.sin(gamma) * np.sin(phi), np.cos(gamma)], axis=-1)

    def extreme(phi, gamma):
        a = _normal_vector(axis(phi, gamma))
        return _davenport_extreme(np.einsum('...i,ij,...j->...', a, l0, a),
                                  np.einsum('...i,ij,...j->...', a, m2, a), T)

    # 1. 1-hat: global maximum of Eq. 17 (grid, then local refinement)
    phi, gamma = np.meshgrid(np.linspace(0, 2*np.pi, 73), np.linspace(0, np.pi, 37), indexing='ij')
    values = extreme(phi, gamma)
    best = np.argmax(values)                      # first maximum, as in the authors' code
    x0 = np.array([phi.flat[best], gamma.flat[best]])
    # no bounds: the angles are periodic, a bound at phi = 0 would stop the refinement there
    res = opt.minimize(lambda x: -extreme(*x), x0, method='SLSQP', options={'ftol': 1e-14})
    x = res.x if -res.fun > values.flat[best] else x0
    n1 = axis(*x)
    n1 = n1 * _matlab_sign(*n1)
    phi = np.arctan2(n1[1], n1[0]) if np.hypot(n1[0], n1[1]) > 0 else 0.
    a_v = np.array([-np.sin(phi), np.cos(phi), 0.])
    a_u = np.cross(n1, a_v)

    # 2. 3-hat = cos(theta) a_u + sin(theta) a_v, 0 <= theta <= pi
    x1, x2 = _max_shear_direction(l0, n1, a_u, a_v)
    n3 = _matlab_sign(0., x2, x1) * (x1 * a_u + x2 * a_v)
    n2 = np.cross(n1, n3)

    # 3. critical plane normal
    delta = 3 * np.pi * (1 - (tau_af / s_af)**2) / 8   # Eq. 19
    w = np.cos(delta) * n1 + np.sin(delta) * n3

    # 4. shear direction on the critical plane, Eq. 20
    k = np.cross(w, n2)
    x1, x2 = _max_shear_direction(l0, w, n2, k)
    v = x1 * n2 + x2 * k

    return voigt_transform(np.stack([np.cross(v, w), v, w]))
