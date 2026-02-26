import numpy as np
import scipy.optimize as opt

def compute_lmn_angles(theta, phi, psi):
    l1 = np.cos(psi) * np.cos(phi) - np.cos(theta) * np.cos(psi) * np.sin(phi)
    m1 = np.sin(psi) * np.cos(phi) + np.cos(theta) * np.cos(psi) * np.sin(phi)
    n1 = np.sin(theta) * np.sin(phi)
    
    l2 = -np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(psi) * np.cos(phi)
    m2 = -np.sin(psi) * np.cos(phi) + np.cos(theta) * np.cos(psi) * np.cos(phi)
    n2 = np.sin(theta) * np.cos(phi)
    
    l3 = np.sin(theta) * np.sin(psi)
    m3 = -np.sin(theta) * np.cos(psi)
    n3 = np.cos(theta)
    
    return l1, m1, n1, l2, m2, n2, l3, m3, n3


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

    
    if 'maxnormal' == method:

        def crit(v):
            """Criterion for max-normal."""
            theta, phi, psi = v
            l1, m1, n1, l2, m2, n2, l3, m3, n3 = compute_lmn_angles(theta, phi, psi)
            a = np.asarray([l1**2, m1**2, n1**2,
                            2*l1*m1, 2*l1*n1, 2*m1*n1])


            return -np.abs(np.einsum('i,ij,j', a, mu, a).real)

        if search_method == 'local':
            res = opt.minimize(crit, [np.pi, np.pi, np.pi], method='SLSQP', bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)])

        elif search_method == 'global':
            res = opt.differential_evolution(crit, bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)])

        else:
            raise ValueError('Search method')


    elif 'maxshear' == method:

        def crit(v):
            """Criterion for max-shear."""
            theta, phi, psi = v
            l1, m1, n1, l2, m2, n2, l3, m3, n3 = compute_lmn_angles(theta, phi, psi)
            a = np.asarray([l1**2 - l3**2, m1**2 - m3**2, n1**2 - n3**2,
                            2 * (l1 * m1 - l3 * m3), 2 * (l1 * n1 - l3 * n3), 2 * (m1 * n1 - m3 * n3)])

            return -np.abs(np.einsum('i,ij,j', a, mu, a).real)

        if search_method == 'local':
            res = opt.minimize(crit, [np.pi, np.pi, np.pi], method='SLSQP', bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)])

        elif search_method == 'global':
            res = opt.differential_evolution(crit, bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)])

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

        if search_method == 'local':
            res = opt.minimize(crit, [np.pi/3, np.pi/3, np.pi/3], method='SLSQP', bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)])

        elif search_method == 'global':
            res = opt.differential_evolution(crit, bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)])

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

    # Step 2: Find 1-hat direction (phi, theta).
    # Maximize Davenport expected extreme of sigma_z at [2,2].
    # C = Tx(theta) @ Tz(phi)  (psi=0 => Tz(psi) = I)
    def s33crit(inp):
        phi, th = inp
        C = Tx(np.cos(th), np.sin(th)) @ Tz(np.cos(phi), np.sin(phi))
        l0_ = C @ l0 @ C.T
        m2_ = C @ m2 @ C.T
        k0 = l0_[2, 2]
        k2 = m2_[2, 2]
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
        return -(C @ l0 @ C.T)[5, 5]

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
        return -(CC @ l0 @ CC.T)[5, 5]

    res = opt.minimize_scalar(tau2crit, 0.33, method='bounded',
                              bounds=(0, 2*np.pi))
    chi = res['x'].real

    CC = (Tz(np.cos(chi), np.sin(chi))
          @ Tx(np.cos(delta), np.sin(delta))
          @ C)
    return CC
