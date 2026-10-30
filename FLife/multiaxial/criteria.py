import warnings
import numpy as np
from scipy.integrate import trapezoid
from . import cplane

def _max_normal(self, s, search_method):
    """
    Internal function for calculating equivalent stress at one node.
    """
    s = cplane._plane_stress_to_voigt_3d(s)
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]
    
    l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.max_variance(multiaxial_psd=s, df=df, method='maxnormal',search_method=search_method)
    
    a = np.asarray([l1**2, m1**2, n1**2,
                2*l1*m1, 2*l1*n1, 2*m1*n1])

    s_eq = np.einsum('i,zij,j', a, s, a)
    return s_eq


def _max_shear(self, s, search_method):
    """
    Internal function for calculating equivalent stress at one node.
    """
    s = cplane._plane_stress_to_voigt_3d(s)
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.max_variance(multiaxial_psd=s, df=df, method='maxshear', search_method=search_method)

    a = np.asarray([l1**2-l3**2, m1**2-m3**2, n1**2-n3**2,
            2*(l1*m1-l3*m3), 2*(l1*n1-l3*n3), 2*(m1*n1-m3*n3)])

    s_eq = np.einsum('i,zij,j', a, s, a)
    return s_eq


def _max_normal_and_shear(self, s, s_af, tau_af, search_method='local'):
    """
    Internal function for calculating equivalent stress at one node.
    """
    s = cplane._plane_stress_to_voigt_3d(s)
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    K = 2 * tau_af / s_af - 1
    l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.max_variance(multiaxial_psd=s, df=df, method='maxshear',K=K, search_method=search_method)

    def plane_vector(e1, e3):
        (l1, m1, n1), (l3, m3, n3) = e1, e3
        return np.asarray([l1**2-l3**2+K*(l1+l3)**2,
                           m1**2-m3**2+K*(m1+m3)**2,
                           n1**2-n3**2+K*(n1+n3)**2,
                           2*(l1*m1-l3*m3+K*(l1+l3)*(m1+m3)),
                           2*(l1*n1-l3*n3+K*(l1+l3)*(n1+n3)),
                           2*(m1*n1-m3*n3+K*(m1+m3)*(n1+n3))]) / (1 + K)

    # The shear stress variance does not change when e3 is reversed or e1 and e3 are swapped, so the
    # search does not tell which of the two maximum-shear planes (Nieslony and Macha 2007, Eq. 3.23) and
    # which shear sign it found. Take the one with the largest equivalent stress variance (Eq. 4.1).
    e1, e3 = np.array([l1, m1, n1]), np.array([l3, m3, n3])
    mu = np.sum(s, axis=0) * df
    a = max((plane_vector(e1, e3), plane_vector(e1, -e3), plane_vector(e3, e1), plane_vector(e3, -e1)),
            key=lambda a: np.einsum('i,ij,j', a, mu, a).real)

    s_eq = np.einsum('i,zij,j', a, s, a)
    return s_eq


def _EVMS(self, s):
    """
    Internal function for calculating equivalent stress at one node.
    """
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    Q = np.array([
        [1,-1/2,-1/2,0,0,0],
        [-1/2,1,-1/2,0,0,0],
        [-1/2,-1/2,1,0,0,0],
        [0,0,0,3,0,0],
        [0,0,0,0,3,0],
        [0,0,0,0,0,3]])
    
    Q_2D = np.array([
        [1,-1/2,0],
        [-1/2,1,0],
        [0,0,3],])

    if s.shape[1] == 6 and s.shape[2] == 6:
        s_eq = np.trace(np.einsum('ij,kjl->kil', Q, s),axis1=1,axis2=2)
    elif s.shape[1] == 3 and s.shape[2] == 3:
        s_eq = np.trace(np.einsum('ij,kjl->kil', Q_2D, s),axis1=1,axis2=2)
    else:
        raise Exception('Input Error. PSD matrix should be the size of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state')
    return s_eq


def _cs(self, s, s_af, tau_af, T=None):
    """
    Internal function for calculating equivalent stress at one node.

    T is the time span for the number of cycles in the Davenport peak factor:
    the given value, else the T given to EquivalentStress or SpectralData, else 1 s.
    """
    s = cplane._plane_stress_to_voigt_3d(s)
    if T is None:
        T = getattr(self, 't', None)
    if T is None:
        T = 1.
        warnings.warn('C-S criterion: observation time T not given, T = 1 s is assumed. '
                      'N1 = nu0*T (Carpinteri et al. 2014, Eqs. 16-17) depends on it, so pass `T`.')

    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    # .. Find critical plane.
    C = cplane.csrandom(multiaxial_psd=s, df=df, s_af=s_af, tau_af=tau_af, f=freq, T=T)
    
    # .. Transform stress.
    s = np.einsum('ij,zjk,lk->zil', C, s, C)
    
    # .. Calculate equivalent PSD.
    sww = s[:, 2, 2]
    svw = s[:, -1, -1,]

    # .. Eq. (22) in Carpinteri et al. (2014): unsquared ratio as published and in the authors' code;
    #    the time-domain criterion (Carpinteri and Spagnoli 2001, Eq. 27) squares it, see issue #51.
    s_eq = sww.real + svw.real * (s_af/tau_af)
    return s_eq


def _multiaxial_rainflow(self, s, k=None):
    """
    Internal function for calculating equivalent stress at one node (f, 3, 3)
    or at all nodes (N, f, 3, 3).

    Pitoiset and Preumont (2000) use one constant unit vector c for the whole spectrum
    (Eqs. 9 and 10). The equivalent PSD is the PSD of s_m = c^T s, c^T Re(S) c (Eqs. 12 and 14).
    c is the combination with the largest damage (Sec. 3), from the single moment method
    (Sec. 4.1). That damage grows with the 2/k-th spectral moment c^T M c, with
    M = int f^(2/k) Re S(f) df, so c is the top eigenvector of M.
    Without k, the exponent is 0 (maximum variance, the limit k -> infinity).
    The vector is stored in `self.multiaxial_rainflow_direction`: (3,) or (N, 3).
    """
    if s.shape[-2:] == (6, 6):
        raise ValueError('Multiaxial rainflow only works for biaxial stresses: PSD matrix (f,3,3)')

    if k is None:
        warnings.warn('multiaxial_rainflow: no S-N slope `k` given. The direction c is chosen for maximum '
                      'variance (k -> infinity), not for maximum damage as in Pitoiset and Preumont (2000). '
                      'Pass `k` for the damage-based choice.', UserWarning, stacklevel=3)
        q = 0.
    else:
        if k <= 0:
            raise ValueError('S-N slope k must be positive.')
        q = 2. / k

    f = np.asarray(self.multiaxial_psd[1], dtype=float)
    # s is Hermitian, so the quadratic form c^T s c only uses the real part.
    r = np.real(s)
    m = trapezoid(f[:, None, None]**q * r, f, axis=-3)
    c = np.linalg.eigh(m)[1][..., -1]
    # c and -c give the same PSD; make the largest component positive for a reproducible c
    idx = np.argmax(np.abs(c), axis=-1)
    c = c * np.sign(np.take_along_axis(c, idx[..., None], axis=-1))
    self.multiaxial_rainflow_direction = c
    return np.einsum('...i,...fij,...j->...f', c, r, c)


def _thermoelastic(self,s):
    """
    Internal function for calculating equivalent stress at one node.
    """
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    Q = np.array([
        [1,1,1,0,0,0],
        [1,1,1,0,0,0],
        [1,1,1,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]])

    Q_2D = np.array([
        [1,1,0],
        [1,1,0],
        [0,0,0]])
    
    if s.shape[1] == 6 and s.shape[2] == 6:
        s_eq = np.trace(np.einsum('ij,kjl->kil', Q, s),axis1=1,axis2=2)
    elif s.shape[1] == 3 and s.shape[2] == 3:
        s_eq = np.trace(np.einsum('ij,kjl->kil', Q_2D, s),axis1=1,axis2=2)
    else:
        raise Exception('Input Error. PSD matrix should be the size of (f, 6, 6) for 3D stress state or (f,3,3) for 2D stress state')
                
    return s_eq


def _liwi(self,s):
    """
    Internal function for calculating equivalent stress at one node.
    """
    s_eq = np.empty((len(s)),dtype=complex)

    for i in range(len(s)):
        sigma_x, sigma_y, tau_xy = s[i,0], s[i,1], s[i,2]

        # Calculate absolute values and angles
        sigma_x_abs = np.abs(sigma_x)
        sigma_y_abs = np.abs(sigma_y)
        tau_xy_abs = np.abs(tau_xy)

        sigma_x_angle = np.angle(sigma_x)
        sigma_y_angle = np.angle(sigma_y)
        tau_xy_angle = np.angle(tau_xy)

        # Define initial parameters
        A1 = sigma_x_abs
        phi_1 = sigma_x_angle
        r = 0

        A2 = np.array([sigma_y_abs, np.sqrt(3) * tau_xy_abs])
        phi_2 = np.array([sigma_y_angle, tau_xy_angle])
        theta = np.array([np.arccos(-1/2), np.pi / 2])

        while r < 2:
            # Calculate intermediate variables in the main loop
            phi_a = np.angle(A1 + A2[r] * np.e**(1j * (phi_2[r] - phi_1 - theta[r])))
            phi_c_overline = theta[r] - 0.5 * np.angle(A2[r]**2 + 2 * A1 * A2[r] * np.cos(phi_1 - phi_2[r]) * np.e**(1j * theta[r]) + A1**2 * np.e**(2j * theta[r]))

            if A2[r] >= A1:
                phi_c = ((phi_c_overline - theta[r] / 2) % (np.pi/2)) + theta[r] / 2
            elif A2[r] < A1:
                phi_c = ((phi_c_overline - theta[r] / 2) % (np.pi/2)) + theta[r] / 2 - np.pi / 2
            else:
                phi_c = phi_c_overline

            A1 = (0.5 * (A1**2 + A2[r]**2 + 2 * A1 * A2[r] * np.cos(phi_1 - phi_2[r]) * np.cos(theta[r])) + 0.5 * np.abs(A1**2 * np.e**(2j * phi_1) + A2[r]**2 * np.e**(2j * phi_2[r]) + 2 * A1 * A2[r] * np.e**(2j * (phi_1 + phi_2[r]))))**0.5
                    
            phi_1 = phi_a + phi_c + phi_1
            r += 1  # Increment r

        # Calculate phi_B0
        phi_B0 = 0.5 * np.angle(sigma_x_abs**2 * np.e**(2j * sigma_x_angle) + sigma_y_abs**2 * np.e**(2j * sigma_y_angle) - sigma_x_abs * sigma_y_abs * np.e**(1j * (sigma_x_angle + sigma_y_angle)) + 3 * tau_xy_abs**2 * np.e**(2j * tau_xy_angle))

        if np.abs(np.angle((np.e**(1j * phi_1))/(np.e**(1j * phi_B0)))) < np.abs(np.angle((np.e**(1j * phi_1))/(np.e**(1j * (phi_B0+np.pi))))):
            phi_vM = phi_B0
        else:
            phi_vM = phi_B0 + np.pi


        # Calculate delta_UM
        sigma_vM_abs = (0.5 * np.abs(sigma_x_abs**2 + sigma_y_abs**2 - sigma_x_abs * sigma_y_abs * np.cos(sigma_x_angle - sigma_y_angle) + 3 * tau_xy_abs**2) + 0.5 * np.abs(sigma_x_abs**2 * np.e**(2j * sigma_x_angle) + sigma_y_abs**2 * np.e**(2j * sigma_y_angle) - sigma_x_abs * sigma_y_abs * np.e**(1j*(sigma_x_angle+sigma_y_angle)) + 3 * tau_xy_abs**2 * np.e**(2j * tau_xy_angle)))**0.5

        sigma_vM = sigma_vM_abs * np.e**(1j * phi_vM)

        s_eq[i] = sigma_vM
    
    return s_eq


def _coin_liwi(self, s, k_a, k_phi):
    """
    Internal function for calculating equivalent stress at one node.
    """
    s_eq = np.empty((len(s)),dtype=complex)

    for i in range(len(s)):

        sigma_x_f = s[i,0]
        sigma_y_f = s[i,1]  
        sigma_z_f = s[i,2]  
        tau_xy_f = s[i,3]
        tau_xz_f = s[i,4]   
        tau_yz_f = s[i,5]   

        S = np.array([[sigma_x_f, tau_xy_f, tau_xz_f],
                    [tau_xy_f, sigma_y_f, tau_yz_f],
                    [tau_xz_f, tau_yz_f, sigma_z_f]])

        I_1_underscore = np.trace(S)

        I_1_squared = np.trace(S) * np.conj(np.trace(S))

        I_1_squared_underscore = np.trace(S) * np.trace(S)

        I_2 = 0.5 * (I_1_squared - np.trace(S@np.conj(S)))

        I_2_underscore = 0.5 * (I_1_squared_underscore - np.trace(S @ S))



        sigma_eq_ka = I_1_squared - k_a**2 * I_2

        sigma_eq_ka_underscore = I_1_squared_underscore - k_a**2 * I_2_underscore

        phi_I1 = np.angle(I_1_underscore)

        phi_eq_ka = np.angle(np.sqrt(sigma_eq_ka_underscore))


        # amplitude
        sigma_coin = np.sqrt(k_phi * sigma_eq_ka + (1 - k_phi) * abs(sigma_eq_ka_underscore))

        #phase angle

        if np.angle(np.exp(1j * phi_I1) / np.exp(1j * phi_eq_ka)) < np.angle(np.exp(1j * phi_I1) / np.exp(1j * phi_eq_ka + np.pi)):
            phi_coin = phi_eq_ka

        else:
            phi_coin = phi_eq_ka + np.pi 

        sigma_coin_f = sigma_coin * np.exp(1j * phi_coin)

        s_eq[i] = sigma_coin_f
    
    return s_eq

def _EVMS_out_of_phase(self, s):
    """
    Internal function for calculating equivalent stress at one node.

    The phases are taken relative to the component with the largest
    auto-PSD at each frequency (Nieslony et al. 2025, Eq. 44). The
    modulus in Bonte et al. (2007), Eq. 9, does not depend on the
    reference component, but a reference with zero PSD has no defined
    phase.
    """
    s_eq = np.empty((len(s)))

    for i in range(len(s)):
        sigma_xx_squared = np.real(s[i,0,0])
        sigma_yy_squared = np.real(s[i,1,1])
        tau_xy_squared = np.real(s[i,2,2])
        sigma_xx_sigma_yy = np.sqrt(sigma_xx_squared * sigma_yy_squared)

        phi_xx_phi_yy = np.angle(s[i,1,0])

        # phases relative to the component with the largest auto-PSD
        c = np.argmax(np.real(np.diagonal(s[i])))
        phi_xx, phi_yy, phi_xy = np.angle(s[i,c,:])

        s_eq[i] = 0.5  * (sigma_xx_squared + sigma_yy_squared - sigma_xx_sigma_yy * np.cos(phi_xx_phi_yy) + 3 * tau_xy_squared) + 0.5 * np.abs(sigma_xx_squared * np.exp(2j * phi_xx) + sigma_yy_squared * np.exp(2j * phi_yy) - sigma_xx_sigma_yy * np.exp(1j * (phi_xx + phi_yy)) + 3*tau_xy_squared * np.exp(2j * phi_xy))

    return s_eq

def _clip_roundoff(G, scale, rtol=1e-10):
    """
    Set round-off negatives of an auto-PSD to 0.

    An auto-PSD is >= 0 in exact arithmetic. Values below 0 but within
    rtol*scale are round-off (e.g. the hydrostatic PSD of a rotated pure shear
    state) and become 0. Larger negative values are kept as they are.
    """
    G = np.real(G)
    return np.where((G < 0) & (G >= -rtol * scale), 0., G)

def _Nieslony(self, s, s_af, tau_af,coefficient_load_type):
    """
    Internal function for calculating equivalent stress at one node.
    """
    s = cplane._plane_stress_to_voigt_3d(s)

    # G_p and G_EMS are auto-PSDs (Niesłony et al. 2025, Eqs. (27)-(30)), so >= 0;
    # clip round-off negatives before sqrt(G_p*G_EMS) in Eq. (31)
    scale = np.real(np.trace(s, axis1=1, axis2=2))
    G_p = _clip_roundoff(1/9*_thermoelastic(self,s), scale)
    G_EVMS = _clip_roundoff(_EVMS(self,s), scale)

    s_eq = np.empty((len(s)))

    for i in range(len(s)): 
        a_sigma = np.sqrt(2*np.diagonal(s[i]))
        #find the index of maximum value of a_sigma
        k = np.argmax(a_sigma)
        phi_k = np.angle(s[i,:,k])
        s_ = a_sigma * np.exp(1j * phi_k)
        sigma_EMSc = 1/np.sqrt(2) * np.sqrt((s_[0]-s_[1])**2 + (s_[1]-s_[2])**2 + (s_[2]-s_[0])**2 + 6*(s_[3]**2+s_[4]**2+s_[5]**2))
        sigma_pc = 1/3 * (s_[0]+s_[1]+s_[2])
        rp_EMS = np.abs(np.cos(np.angle(sigma_EMSc)-np.angle(sigma_pc)))

        if coefficient_load_type == 'tension':
            A = (-np.sqrt(3) / tau_af) * ((s_af * rp_EMS) - np.sqrt((s_af ** 2) * (rp_EMS ** 2) - (s_af ** 2) + 3 * (tau_af ** 2)))
            B = (np.sqrt(3) / 3) * (s_af / tau_af)

        elif coefficient_load_type == 'torsion':
            A = (-np.sqrt(3) / s_af) * ((s_af * rp_EMS) - np.sqrt((s_af ** 2) * (rp_EMS ** 2) - (s_af ** 2) + 3 * (tau_af ** 2)))
            B = np.sqrt(3) / 3
        else:
            raise ValueError('Invalid coefficient_load_type. Please choose either "tension" or "torsion"')
        
        s_eq[i] = np.real(A**2 * G_p[i] + B**2 * G_EVMS[i] + 2 * A * B * rp_EMS * np.sqrt(G_p[i] * G_EVMS[i]))
    
    return s_eq

def _Lemaitre(self, s, poisson_ratio):
    """
    Internal function for calculating equivalent stress at one node.
    """
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    Q = np.array([
        [1,-poisson_ratio,-poisson_ratio,0,0,0],
        [-poisson_ratio,1,-poisson_ratio,0,0,0],
        [-poisson_ratio,-poisson_ratio,1,0,0,0],
        [0,0,0,2*(1+poisson_ratio),0,0],
        [0,0,0,0,2*(1+poisson_ratio),0],
        [0,0,0,0,0,2*(1+poisson_ratio)]])
    
    Q_2D = np.array([
        [1,-poisson_ratio,0],
        [-poisson_ratio,1,0],
        [0,0,2*(1+poisson_ratio)],])

    if s.shape[1] == 6 and s.shape[2] == 6:
        s_eq = np.trace(np.einsum('ij,kjl->kil', Q, s),axis1=1,axis2=2)
    elif s.shape[1] == 3 and s.shape[2] == 3:
        s_eq = np.trace(np.einsum('ij,kjl->kil', Q_2D, s),axis1=1,axis2=2)
    else:
        raise Exception('Input Error. PSD matrix should be the size of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state')
    return s_eq
