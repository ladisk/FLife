import warnings
import numpy as np
from scipy.integrate import trapezoid
from . import cplane


def _quadratic_criterion(s, Q, Q_2D):
    """
    trace(Q S) for a constant weight matrix Q, at one node (f, n, n) or at all nodes (N, f, n, n).

    S is Hermitian and Q real symmetric, so trace(Q S) is real; the round-off imaginary part is dropped.
    """
    if s.shape[-2:] == (6, 6):
        return np.real(np.einsum('ij,...fji->...f', Q, s))
    elif s.shape[-2:] == (3, 3):
        return np.real(np.einsum('ij,...fji->...f', Q_2D, s))
    raise Exception('Input Error. PSD matrix should be the size of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state')


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

    # s is Hermitian and a real, so a^T s a is real; drop the round-off imaginary part
    s_eq = np.real(np.einsum('i,zij,j', a, s, a))
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

    # s is Hermitian and a real, so a^T s a is real; drop the round-off imaginary part
    s_eq = np.real(np.einsum('i,zij,j', a, s, a))
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

    # s is Hermitian and a real, so a^T s a is real; drop the round-off imaginary part
    s_eq = np.real(np.einsum('i,zij,j', a, s, a))
    return s_eq


def _gao_mwcm(self, s, s_af, tau_af, search_method='local'):
    """
    Internal function for calculating equivalent stress at one node.

    Gao et al. (2021), zero mean stress (p = 1): tau_eq = tau_c + q sigma_c on the plane of
    maximum shear stress variance (Eq. 12). G_eq = a^T G a with a = d1 + q d2 (Eqs. 20, 25, 27).
    q = 1 - s_af / (2 tau_af) from Susmel and Lazzarin (2002), Eq. 11; Gao's Eq. 14 has
    sigma_-1 and tau_-1 swapped.
    """
    s = cplane._plane_stress_to_voigt_3d(s)

    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    q = 1 - s_af / (2 * tau_af)
    l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.max_variance(multiaxial_psd=s, df=df, method='maxshear', search_method=search_method)

    def plane_vector(e1, e3):
        # normal n = (e1 + e3)/sqrt2 and shear direction t = (e1 - e3)/sqrt2:
        # d1 = Voigt(t n^T + n t^T)/2 -> (e1 e1^T - e3 e3^T)/2 (Gao Eq. 8),
        # d2 = Voigt(n n^T) -> (e1 + e3)(e1 + e3)^T/2 (Gao Eq. 10)
        (l1, m1, n1), (l3, m3, n3) = e1, e3
        return np.asarray([l1**2-l3**2+q*(l1+l3)**2,
                           m1**2-m3**2+q*(m1+m3)**2,
                           n1**2-n3**2+q*(n1+n3)**2,
                           2*(l1*m1-l3*m3+q*(l1+l3)*(m1+m3)),
                           2*(l1*n1-l3*n3+q*(l1+l3)*(n1+n3)),
                           2*(m1*n1-m3*n3+q*(m1+m3)*(n1+n3))]) / 2

    # Gao Eq. 12 maximises the shear stress variance only. That variance does not change when e3 is
    # reversed or e1 and e3 are swapped, so it does not tell which of the two maximum-shear planes and
    # which shear sign to use. Our choice (not from the article), as in _max_normal_and_shear: take the
    # one with the largest equivalent stress variance.
    e1, e3 = np.array([l1, m1, n1]), np.array([l3, m3, n3])
    mu = np.sum(s, axis=0) * df
    a = max((plane_vector(e1, e3), plane_vector(e1, -e3), plane_vector(e3, e1), plane_vector(e3, -e1)),
            key=lambda a: np.einsum('i,ij,j', a, mu, a).real)

    # G_eq = G_tau + 2 q Re(G_tau,sigma) + q^2 G_sigma (Gao Eq. 25 with p = 1).
    # s is Hermitian and a real, so a^T s a is real; drop the round-off imaginary part
    s_eq = np.real(np.einsum('i,zij,j', a, s, a))
    # tau_eq is a shear stress; scale it by s_af / tau_af to an equivalent normal stress, as the
    # other criteria return, so that it is used with the axial S-N curve
    return s_eq * (s_af / tau_af)**2


def _EVMS(self, s):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.
    """
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

    return _quadratic_criterion(s, Q, Q_2D)


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
    Internal function for calculating equivalent stress at one node or at all nodes.
    """
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

    return _quadratic_criterion(s, Q, Q_2D)


def _EVMS_out_of_phase(self, s):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.

    The phases are taken relative to the component with the largest
    auto-PSD at each frequency (Nieslony et al. 2025, Eq. 44). The
    modulus in Bonte et al. (2007), Eq. 9, does not depend on the
    reference component, but a reference with zero PSD has no defined
    phase.
    """
    if s.shape[-2:] != (3, 3):
        raise ValueError('EVMS_out_of_phase only works for biaxial stresses: PSD matrix (f,3,3) or (N,f,3,3)')

    sigma_xx_squared = np.real(s[...,0,0])
    sigma_yy_squared = np.real(s[...,1,1])
    tau_xy_squared = np.real(s[...,2,2])
    sigma_xx_sigma_yy = np.sqrt(sigma_xx_squared * sigma_yy_squared)

    phi_xx_phi_yy = np.angle(s[...,1,0])

    # phases relative to the component with the largest auto-PSD
    c = np.argmax(np.real(np.diagonal(s, axis1=-2, axis2=-1)), axis=-1)
    phi = np.angle(np.take_along_axis(s, c[..., None, None], axis=-2)[..., 0, :])
    phi_xx, phi_yy, phi_xy = phi[...,0], phi[...,1], phi[...,2]

    s_eq = 0.5  * (sigma_xx_squared + sigma_yy_squared - sigma_xx_sigma_yy * np.cos(phi_xx_phi_yy) + 3 * tau_xy_squared) + 0.5 * np.abs(sigma_xx_squared * np.exp(2j * phi_xx) + sigma_yy_squared * np.exp(2j * phi_yy) - sigma_xx_sigma_yy * np.exp(1j * (phi_xx + phi_yy)) + 3*tau_xy_squared * np.exp(2j * phi_xy))

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
    Internal function for calculating equivalent stress at one node or at all nodes.
    """
    if coefficient_load_type not in ('tension', 'torsion'):
        raise ValueError('Invalid coefficient_load_type. Please choose either "tension" or "torsion"')

    s = cplane._plane_stress_to_voigt_3d(s)

    # G_p and G_EMS are auto-PSDs (Niesłony et al. 2025, Eqs. (27)-(30)), so >= 0;
    # clip round-off negatives before sqrt(G_p*G_EMS) in Eq. (31)
    scale = np.real(np.trace(s, axis1=-2, axis2=-1))
    G_p = _clip_roundoff(1/9*_thermoelastic(self,s), scale)
    G_EVMS = _clip_roundoff(_EVMS(self,s), scale)

    a_sigma = np.sqrt(2*np.diagonal(s, axis1=-2, axis2=-1))
    #find the index of maximum value of a_sigma
    k = np.argmax(a_sigma, axis=-1)
    phi_k = np.angle(np.take_along_axis(s, k[..., None, None], axis=-1)[..., 0])
    s_ = a_sigma * np.exp(1j * phi_k)
    s_ = [s_[..., i] for i in range(6)]
    sigma_EMSc = 1/np.sqrt(2) * np.sqrt((s_[0]-s_[1])**2 + (s_[1]-s_[2])**2 + (s_[2]-s_[0])**2 + 6*(s_[3]**2+s_[4]**2+s_[5]**2))
    sigma_pc = 1/3 * (s_[0]+s_[1]+s_[2])
    rp_EMS = np.abs(np.cos(np.angle(sigma_EMSc)-np.angle(sigma_pc)))

    if coefficient_load_type == 'tension':
        A = (-np.sqrt(3) / tau_af) * ((s_af * rp_EMS) - np.sqrt((s_af ** 2) * (rp_EMS ** 2) - (s_af ** 2) + 3 * (tau_af ** 2)))
        B = (np.sqrt(3) / 3) * (s_af / tau_af)

    elif coefficient_load_type == 'torsion':
        A = (-np.sqrt(3) / s_af) * ((s_af * rp_EMS) - np.sqrt((s_af ** 2) * (rp_EMS ** 2) - (s_af ** 2) + 3 * (tau_af ** 2)))
        B = np.sqrt(3) / 3

    s_eq = np.real(A**2 * G_p + B**2 * G_EVMS + 2 * A * B * rp_EMS * np.sqrt(G_p * G_EVMS))

    return s_eq

def _Lemaitre(self, s, poisson_ratio):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.
    """
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

    return _quadratic_criterion(s, Q, Q_2D)
