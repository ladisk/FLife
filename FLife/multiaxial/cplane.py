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


def _maximize(crit, x0, bounds, search_method, n_starts=16):
    """Maximize -crit over the plane orientation.

    'local' uses a deterministic multi-start SLSQP (the default start x0 plus
    seeded random starts) so the search is not trapped at a stationary start
    point; 'global' uses differential evolution. Results are reproducible.
    """
    if search_method == 'global':
        return opt.differential_evolution(crit, bounds=bounds, seed=0)
    elif search_method == 'local':
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        starts = [np.asarray(x0, dtype=float)]
        starts += [rng.uniform(lo, hi) for _ in range(n_starts - 1)]
        best = None
        for s0 in starts:
            res = opt.minimize(crit, s0, method='SLSQP', bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        return best
    else:
        raise ValueError('Search method')


def max_variance_old(multiaxial_psd, df, method, K=None):
    '''Determine critical plane
    based on max variance.

    Notes
    -----
    Took this from article (its older tough):
        - Lagoda, Macha and Nieslony; Fatigue Life Calculation
        by means of the cycle counting and spectral methods
        under multiaxial random loading.
    '''
    mu = np.sum(multiaxial_psd, axis=0) * df

    if 'maxnormal' == method:

        def crit(v):
            """Criterion for optimization."""
            v = v / np.sqrt(np.sum(v ** 2))
            l, m, n = v
            a = np.asarray([l**2, m**2, n**2,
                            2*l*m, 2*l*n, 2*m*n])

            return -np.abs(np.einsum('i,ij,j', a, mu, a).real)


        res = opt.minimize(fun=crit, x0=[0.33,0.33,0.33], method='SLSQP', bounds=[(0,1),(0,1),(0,1)], options={'disp':False})

    return res['x'] / np.sqrt(np.sum(res['x']**2))


def max_variance(multiaxial_psd, df, method, K=None, search_method='local'):
    '''Determine critical plane
    based on max variance.

    returns: l1, m1, n1, l2, m2, n2, l3, m3, n3
    
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

        res = _maximize(crit, [np.pi, np.pi, np.pi], bounds, search_method)


    elif 'maxshear' == method:

        def crit(v):
            """Criterion for max-shear."""
            theta, phi, psi = v
            l1, m1, n1, l2, m2, n2, l3, m3, n3 = compute_lmn_angles(theta, phi, psi)
            a = np.asarray([l1**2 - l3**2, m1**2 - m3**2, n1**2 - n3**2,
                            2 * (l1 * m1 - l3 * m3), 2 * (l1 * n1 - l3 * n3), 2 * (m1 * n1 - m3 * n3)])

            return -np.abs(np.einsum('i,ij,j', a, mu, a).real)

        res = _maximize(crit, [np.pi, np.pi, np.pi], bounds, search_method)

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

        res = _maximize(crit, [np.pi/3, np.pi/3, np.pi/3], bounds, search_method)

    else:
        raise ValueError('Critical plane method')

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


def _davenport_extreme(k0, k2):
    """Davenport expected maximum sqrt(k0) * (sqrt(2 ln N1) + 0.5772 / sqrt(2 ln N1)), Eq. 17.

    N1 = sqrt(k2 / k0) / (2 pi) for moments k in Hz. The value is 0 where N1 <= 1
    (the formula is undefined there).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        N1 = np.sqrt(np.where(k0 > 0, k2 / k0, 0.)) / (2 * np.pi)
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


def csrandom(multiaxial_psd, df, s_af, tau_af, f=None):
    '''Determine critical plane with the C-S random criterion.

    Returns the 6x6 Voigt transformation CC to the critical-plane frame:
    after transformation, sigma_w = S[2,2] and tau_vw = S[5,5].

    :param f: frequency vector in Hz for the spectral moments (Eq. 15a/b);
        defaults to arange(len) * df

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
                                  np.einsum('...i,ij,...j->...', a, m2, a))

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
