import numpy as np
import scipy.optimize as opt

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
    mu = np.sum(multiaxial_psd, axis=0) * df
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
# The C-S algorithm uses ZXZ Euler angles (phi, theta, psi):
#   C_phi  = Tz(phi)   -- rotation about Z
#   C_theta = Tx(theta) -- rotation about X (line of nodes)
#   C_psi  = Tz(psi)   -- rotation about Z' (leaves sigma_z invariant)
# Plus two more rotations for the critical plane:
#   C_delta = Tx(delta) -- rotation about X' (2-hat axis)
#   C_chi   = Tz(chi)   -- rotation about Z (w-axis, leaves sigma_w invariant)

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


def csrandom(multiaxial_psd, df, s_af, tau_af):
    '''Determine critical plane with the C-S random criterion.

    Uses ZXZ Euler angles with 6x6 Voigt stress transformations derived
    from first principles. The 5-angle sequence is:
        phi (about Z), theta (about X), psi (about Z'),
        delta (about X', material angle), chi (about Z/w-axis).

    After transformation, sigma_w = S[2,2] and tau_vw = S[5,5].

    See also
    --------
    Carpinteri A, Spagnoli A and Vantadori S, Reformulation in the frequency
    domain of a critical plane-based multiaxial fatigue criterion,
    Int J Fat, 2014
    '''
    f = np.arange(0, multiaxial_psd.shape[0] * df, df)
    l0 = spectral_moment(f, multiaxial_psd, 0)
    m2 = spectral_moment(f, multiaxial_psd, 2)

    # Normalise the moments before the plane search. The critical plane depends
    # only on the relative stress distribution, but SLSQP uses absolute
    # tolerances, so an objective that scales with the PSD magnitude converges to
    # a slightly different plane at different load levels. Scaling l0 and m2 by a
    # common factor keeps the objective O(1) and leaves both the plane and the
    # moment ratio in N1 unchanged. The equivalent stress is computed later in
    # _cs from the unscaled stress.
    scale = np.real(np.trace(l0))
    if scale > 0:
        l0 = l0 / scale
        m2 = m2 / scale

    # Step 2: Find 1-hat direction (phi, theta).
    # Maximize Davenport expected extreme of sigma_z at [2,2].
    # C = Tx(theta) @ Tz(phi)  (psi=0 => Tz(psi) = I)
    def s33crit(inp):
        phi, th = inp
        C = Tx(np.cos(th), np.sin(th)) @ Tz(np.cos(phi), np.sin(phi))
        l0_ = C @ l0 @ C.T
        m2_ = C @ m2 @ C.T
        k0 = np.real(l0_[2, 2])
        k2 = np.real(m2_[2, 2])
        if k0 <= 0 or k2 <= 0:
            return 0.0
        N1 = np.sqrt(k2 / k0) / (2 * np.pi)
        if N1 <= 1:
            return 0.0
        return -(np.sqrt(k0) * (np.sqrt(2 * np.log(N1))
                 + 0.5772 / np.sqrt(2 * np.log(N1))))

    res = opt.minimize(s33crit, [0.33, 0.33], method='SLSQP',
                       bounds=[(0, 2*np.pi), (0, np.pi)])
    phi, th = res['x']

    # Step 3: Find 3-hat direction (psi).
    # Maximize variance of tau_yz at [5,5].
    # Tz(psi) leaves sigma_z invariant, so normal stress is preserved.
    def taucrit(psi):
        C = (Tz(np.cos(psi), np.sin(psi))
             @ Tx(np.cos(th), np.sin(th))
             @ Tz(np.cos(phi), np.sin(phi)))
        return -np.real((C @ l0 @ C.T)[5, 5])

    res = opt.minimize_scalar(taucrit, 0.33, method='bounded',
                              bounds=(0, 2*np.pi))
    psi = res['x'].real

    # Step 4: Off-angle delta (Eq. 19).
    delta = 3 * np.pi * (1 - (tau_af / s_af)**2) / 8

    # Build 3-angle rotation (phi, theta, psi).
    C = (Tz(np.cos(psi), np.sin(psi))
         @ Tx(np.cos(th), np.sin(th))
         @ Tz(np.cos(phi), np.sin(phi)))

    # Step 5: Find chi — maximize variance of tau_vw at [5,5].
    # CC = Tz(chi) @ Tx(delta) @ C
    # Tz(chi) leaves sigma_w invariant after delta tilt.
    def tau2crit(chi):
        CC = (Tz(np.cos(chi), np.sin(chi))
              @ Tx(np.cos(delta), np.sin(delta))
              @ C)
        return -np.real((CC @ l0 @ CC.T)[5, 5])

    res = opt.minimize_scalar(tau2crit, 0.33, method='bounded',
                              bounds=(0, 2*np.pi))
    chi = res['x'].real

    CC = (Tz(np.cos(chi), np.sin(chi))
          @ Tx(np.cos(delta), np.sin(delta))
          @ C)
    return CC
