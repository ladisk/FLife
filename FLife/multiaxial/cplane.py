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
            res = opt.minimize(crit, [np.pi, np.pi, np.pi], method='L-BFGS-B', bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)], options={'disp':False})
        
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
            res = opt.minimize(crit, [np.pi, np.pi, np.pi], method='L-BFGS-B', bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)], options={'disp': False})
        
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
            res = opt.minimize(crit, [np.pi/3, np.pi/3, np.pi/3], method='L-BFGS-B', bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)], options={'disp': False})
        
        elif search_method == 'global':
            res = opt.differential_evolution(crit, bounds=[(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)])

    else:
        raise ValueError('Critical plane method')

    direction_cosines = compute_lmn_angles(*res['x'])
    
    return direction_cosines


# Carpinteri-Spagnoli

def C1rot(cpsi, spsi):
    '''Rotation matrix for angle psi.'''
    return np.array([[1, 0, 0, 0, 0, 0],
                     [0, cpsi**2, spsi**2, 2*cpsi*spsi, 0, 0],
                     [0, spsi**2, cpsi**2, -2*cpsi*spsi, 0, 0],
                     [0, -2*cpsi*spsi, 2*cpsi*spsi, cpsi**2-spsi**2, 0, 0],
                     [0, 0, 0, 0, cpsi, -spsi],
                     [0, 0, 0, 0, spsi, -cpsi]])

def C2rot(cth, sth):
    '''Rotation matrix for angle theta.'''
    return np.array([[cth**2, 0, sth**2, 0, 2*cth*sth, 0],
                     [0, 1, 0, 0, 0, 0],
                     [sth**2, 0, cth**2, 0, -2*cth*sth, 0],
                     [0, 0, 0, cth, 0, -sth],
                     [-cth*sth, 0, sth*sth, 0, cth**2-sth**2, 0],
                     [0, 0, 0, sth, 0, cth]])

def C3rot(cphi, sphi):
    '''Rotation matrix for angle phi.'''
    return np.array([[1, 0, 0, 0, 0, 0],
                     [0, cphi**2, sphi**2, 2*cphi*sphi, 0, 0],
                     [0, sphi**2, cphi**2, -2*cphi*sphi, 0, 0],
                     [0, -2*cphi*sphi, 2*cphi*sphi, cphi**2-sphi**2, 0, 0],
                     [0, 0, 0, 0, cphi, -sphi],
                     [0, 0, 0, 0, sphi, cphi]])

def C4rot(cgam, sgam):
    '''Rotation matrix for angle gamma.'''
    return np.array([[cgam**2, sgam**2, 0, 0, 0, 2*cgam*sgam],
                     [sgam**2, cgam**2, 0, 0, 0, -2*cgam*sgam],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, cgam, sgam, 0],
                     [0, 0, 0, -sgam, cgam, 0],
                     [-cgam*sgam, cgam*sgam, 0, 0, 0, cgam**2-sgam**2]])

def C5rot(cdel, sdel):
    '''Rotation matrix for angle delta.'''
    return np.array([[1, 0, 0, 0, 0, 0],
                     [0, cdel**2, sdel**2, -2*cdel*sdel, 0, 0],
                     [0, sdel**2, cdel**2, 2*cdel*sdel, 0, 0],
                     [0, cdel*sdel, -cdel*sdel, 0, 0, 0],
                     [0, 0, 0, 0, cdel, sdel],
                     [0, 0, 0, 0, -sdel, cdel]])


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
    '''Determine critical plane with
    the C-S random criterion.
    
    See also
    --------
    Carpinteri A, Spagnoli A and Vantadori S, Reformulation in the frequency domain of a critical
    plane-based multiaxial fatigue criterion, Int J Fat, 2014
    '''

    f = np.arange(0, multiaxial_psd.shape[0]*df, df)
    l0 = spectral_moment(f, multiaxial_psd, 0)
    N1 = np.sqrt(spectral_moment(f, multiaxial_psd, 2) / l0) / (2*np.pi)
    
    # -- S33 first; extreme according to Davenport.
    # .. Varying angles 0 < phi < 2pi and 0 < theta < pi
    def s33crit(inp):
        """Optimisation criterion (searching
        for maximum)."""
        phi, th = inp
        C = np.dot(C2rot(np.cos(th), np.sin(th)), C3rot(np.cos(phi), np.sin(phi)))
        l0_ = np.dot(np.dot(C, l0), C.transpose())
        N1_ = np.dot(np.dot(C, N1), C.transpose())
        
        return -(np.sqrt(l0_[2,2]) * np.sqrt(2*np.log(N1_[2,2]))
                 + 0.5772/np.sqrt(2*np.log(N1_[2,2])))
        
    res = opt.minimize(s33crit, [0.33,0.33], method='SLSQP', bounds=[(0,2*np.pi),(0,np.pi)], options={'disp':False})

    #res = opt.differential_evolution(s33crit, bounds=[(0,2*np.pi),(0,np.pi)])

    


    phi, th = res['x']
    
    # -- Tau_yz second; maximum variance.
    def taucrit(psi):
        '''Optimisation criterion (searching for
        maximum variance (mu).'''
        C = np.dot(
                   np.dot(C1rot(np.cos(psi), np.sin(psi)), C2rot(np.cos(th), np.sin(th))),
                   C3rot(np.cos(phi), np.sin(phi)))
        
        return -np.dot(np.dot(C, l0), C.transpose())[-1, -1]
        
    
    res = opt.minimize_scalar(taucrit, 0.33, method='bounded', bounds=(0,2*np.pi), options={'disp':False})
    #res = opt.differential_evolution(taucrit, bounds=(0,2*np.pi))
    psi = res['x'].real
    

    delta = 3*np.pi*(1-(tau_af)**2/(s_af)**2)/8

    # -- Maximize shear variance
    C = np.dot(
               np.dot(C1rot(np.cos(psi), np.sin(psi)), C2rot(np.cos(th), np.sin(th))),
               C3rot(np.cos(phi), np.sin(phi)))
    def tau2crit(gam):
        '''Optimisation criterion (searching for
        maximum variance).'''
        CC = np.dot(
                    np.dot(C4rot(np.cos(gam), np.sin(gam)), C5rot(np.cos(delta), np.sin(delta))), C)
        
        return -np.dot(np.dot(CC, l0), C.transpose())[-1, -1]
    
    res = opt.minimize_scalar(tau2crit, 0.33, method='bounded', bounds=(0, 2*np.pi), options={'disp':False})
    #res = opt.differential_evolution(tau2crit, bounds=(0,2*np.pi))
    gam = res['x'].real
    
    CC = np.dot(
            np.dot(C4rot(np.cos(gam), np.sin(gam)), C5rot(np.cos(delta), np.sin(delta))), C)
    
    return CC



