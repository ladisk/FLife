import warnings
import numpy as np
from . import cplane

# All criteria accept the PSD of one node (f, 6, 6) / (f, 3, 3) or of many
# nodes (N, f, 6, 6) / (N, f, 3, 3) and return the equivalent PSD (f,) or (N, f).


def _df(self):
    freq = self._freq()
    return freq[1] - freq[0]


def _psd_along(a, s):
    """PSD of the scalar a . sigma, a (..., 6) and s (..., f, 6, 6)."""
    return np.real(np.einsum('...i,...fij,...j->...f', a, s, a))


def _plane_vector(frames, K=None):
    """Voigt coefficients of the critical-plane stress for direction cosines (..., 3, 3).

    K=None gives the normal stress on eta_1, otherwise
    (sigma_11 - sigma_33 + K (eta_1 + eta_3)^T sigma (eta_1 + eta_3)) / (1 + K),
    which is 2 tau for K=0.
    """
    e1, e3 = frames[..., 0, :], frames[..., 2, :]
    if K is None:
        return cplane._normal_vector(e1)
    return (cplane._normal_vector(e1) - cplane._normal_vector(e3)
            + K * cplane._normal_vector(e1 + e3)) / (1 + K)


def _quadratic_criterion(s, Q, Q_2D):
    """trace(Q S) for a constant weight matrix Q."""
    if s.shape[-2:] == (6, 6):
        return np.real(np.einsum('ij,...fji->...f', Q, s))
    elif s.shape[-2:] == (3, 3):
        return np.real(np.einsum('ij,...fji->...f', Q_2D, s))
    raise Exception('Input Error. PSD matrix should be the size of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state')


def _max_normal(self, s, search_method):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.
    """
    s = cplane._plane_stress_to_voigt_3d(s)
    frames = cplane.max_variance_planes(s, _df(self), 'maxnormal', search_method=search_method)
    return _psd_along(_plane_vector(frames), s)


def _max_shear(self, s, search_method):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.
    """
    s = cplane._plane_stress_to_voigt_3d(s)
    frames = cplane.max_variance_planes(s, _df(self), 'maxshear', search_method=search_method)
    return _psd_along(_plane_vector(frames, K=0.), s)


def _max_normal_and_shear(self, s, s_af, tau_af, search_method='local'):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.

    The critical plane maximises the variance of the equivalent stress itself
    (Bedkowski and Macha), so the result does not depend on how the plane
    search labels eta_1 and eta_3.
    """
    s = cplane._plane_stress_to_voigt_3d(s)
    K = 2 * tau_af / s_af - 1
    frames = cplane.max_variance_planes(s, _df(self), 'maxnormalshear', K=K, search_method=search_method)
    return _psd_along(_plane_vector(frames, K=K), s)


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
    Internal function for calculating equivalent stress at one node or at all nodes.

    T is the time span for the number of cycles in the Davenport peak factor:
    the given value, else the T given to EquivalentStress or SpectralData.
    """
    s = cplane._plane_stress_to_voigt_3d(s)
    if T is None:
        T = getattr(self, 't', None)
    if T is None:
        raise ValueError('C-S criterion needs the observation time T: pass `T` to `cs`, or `T` and `fs` to EquivalentStress.')

    # .. Find critical plane: rows (u, v, w), w is the plane normal.
    frames = cplane.cs_planes(s, self._freq(), s_af=s_af, tau_af=tau_af, T=T)
    v, w = frames[..., 1, :], frames[..., 2, :]

    # .. Calculate equivalent PSD.
    sww = _psd_along(cplane._normal_vector(w), s)
    svw = _psd_along(cplane._shear_vector(v, w), s)

    s_eq = sww + svw * (s_af/tau_af)
    return s_eq


def _multiaxial_rainflow(self, s, k=None):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.

    One fixed unit vector c is used for the whole spectrum (Pitoiset and Preumont 2000, Eq. 9).
    The equivalent PSD is c^T Re(S) c (Eq. 14). c is the top eigenvector of
    M = sum_f f^(2/k) Re S(f) df, the maximum of the single-moment damage over c.
    Without k, the exponent is 0 (maximum variance, the limit k -> infinity).
    The vector is stored in `self.multiaxial_rainflow_direction`: (3,) or (N, 3).
    """
    if s.shape[-2:] == (6, 6):
        raise ValueError('Multiaxial rainflow only works for biaxial stresses: PSD matrix (f,3,3)')

    f = np.asarray(self._freq(), dtype=float)
    if k is None:
        warnings.warn('multiaxial_rainflow: no S-N slope `k` given. The direction c is chosen for maximum '
                      'variance (k -> infinity), not for maximum damage as in Pitoiset and Preumont (2000). '
                      'Pass `k` for the damage-based choice.', UserWarning, stacklevel=3)
        q = 0.
    else:
        if k <= 0:
            raise ValueError('S-N slope k must be positive.')
        q = 2. / k

    r = np.real(s)
    weight = np.where(f > 0, np.abs(f), 0.)**q if q > 0 else np.ones_like(f)
    weight = weight * _df(self)
    m = np.einsum('f,...fij->...ij', weight, r)
    c = np.linalg.eigh(m)[1][..., -1]
    # fix the sign: largest component positive
    idx = np.argmax(np.abs(c), axis=-1)
    sign = np.sign(np.take_along_axis(c, idx[..., None], axis=-1))
    c = c * np.where(sign == 0, 1., sign)
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


def _liwi(self,s):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.

    s is the amplitude spectrum (..., f, 3); returns the complex equivalent amplitude (..., f).
    Schmidt & Pandiya (2021): amplitude from Eqs. 10-11 (Bonte), equal to COIN-LiWI with k_a**2 = 3 and
    k_phi = 0.5; phase from Eqs. 12, 15-18, 27-30.
    Note: PSD = |A|**2/df, so A is the RMS amplitude per line; for one-sided peak amplitudes (FFT amplitudes,
    Schmidt and Pandiya 2021, Eqs. 1 and 7) pass A/sqrt(2).
    """
    sigma_x, sigma_y, tau_xy = s[...,0], s[...,1], s[...,2]

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

    A2 = [sigma_y_abs, np.sqrt(3) * tau_xy_abs]
    phi_2 = [sigma_y_angle, tau_xy_angle]
    theta = [np.arccos(-1/2), np.pi / 2]

    for r in range(2):
        # Calculate intermediate variables in the main loop
        phi_a = np.angle(A1 + A2[r] * np.e**(1j * (phi_2[r] - phi_1 - theta[r])))
        phi_c_overline = theta[r] - 0.5 * np.angle(A2[r]**2 + 2 * A1 * A2[r] * np.cos(phi_1 - phi_2[r]) * np.e**(1j * theta[r]) + A1**2 * np.e**(2j * theta[r]))

        phi_c = ((phi_c_overline - theta[r] / 2) % (np.pi/2)) + theta[r] / 2
        phi_c = np.where(A2[r] >= A1, phi_c, phi_c - np.pi / 2)

        # amplitude of the resultant vector (Schmidt & Pandiya 2021, Eqs. 15, 17, 22, 24): cross term 2 A1 A2 cos(theta) e^{j(phi_1+phi_2)}
        A1 = (0.5 * (A1**2 + A2[r]**2 + 2 * A1 * A2[r] * np.cos(phi_1 - phi_2[r]) * np.cos(theta[r])) + 0.5 * np.abs(A1**2 * np.e**(2j * phi_1) + A2[r]**2 * np.e**(2j * phi_2[r]) + 2 * A1 * A2[r] * np.cos(theta[r]) * np.e**(1j * (phi_1 + phi_2[r]))))**0.5

        phi_1 = phi_a + phi_c + phi_1

    # Calculate phi_B0
    phi_B0 = 0.5 * np.angle(sigma_x_abs**2 * np.e**(2j * sigma_x_angle) + sigma_y_abs**2 * np.e**(2j * sigma_y_angle) - sigma_x_abs * sigma_y_abs * np.e**(1j * (sigma_x_angle + sigma_y_angle)) + 3 * tau_xy_abs**2 * np.e**(2j * tau_xy_angle))

    closer_to_B0 = np.abs(np.angle((np.e**(1j * phi_1))/(np.e**(1j * phi_B0)))) < np.abs(np.angle((np.e**(1j * phi_1))/(np.e**(1j * (phi_B0+np.pi)))))
    phi_vM = np.where(closer_to_B0, phi_B0, phi_B0 + np.pi)

    # Calculate delta_UM
    sigma_vM_abs = (0.5 * np.abs(sigma_x_abs**2 + sigma_y_abs**2 - sigma_x_abs * sigma_y_abs * np.cos(sigma_x_angle - sigma_y_angle) + 3 * tau_xy_abs**2) + 0.5 * np.abs(sigma_x_abs**2 * np.e**(2j * sigma_x_angle) + sigma_y_abs**2 * np.e**(2j * sigma_y_angle) - sigma_x_abs * sigma_y_abs * np.e**(1j*(sigma_x_angle+sigma_y_angle)) + 3 * tau_xy_abs**2 * np.e**(2j * tau_xy_angle)))**0.5

    return sigma_vM_abs * np.e**(1j * phi_vM)


def _coin_liwi(self, s, k_a, k_phi):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.

    s is the amplitude spectrum (..., f, 6); returns the complex equivalent amplitude (..., f).
    Note: PSD = |A|**2/df, so A is the RMS amplitude per line; for one-sided peak amplitudes (FFT amplitudes,
    Schmidt and Kraft 2023) pass A/sqrt(2).
    """
    sigma_x_f = s[...,0]
    sigma_y_f = s[...,1]
    sigma_z_f = s[...,2]
    tau_xy_f = s[...,3]
    tau_xz_f = s[...,4]
    tau_yz_f = s[...,5]

    # S is symmetric: trace(S conj(S)) and trace(S S) from the components
    I_1_underscore = sigma_x_f + sigma_y_f + sigma_z_f

    I_1_squared = I_1_underscore * np.conj(I_1_underscore)

    I_1_squared_underscore = I_1_underscore * I_1_underscore

    trace_S_conjS = (np.abs(sigma_x_f)**2 + np.abs(sigma_y_f)**2 + np.abs(sigma_z_f)**2
                     + 2 * (np.abs(tau_xy_f)**2 + np.abs(tau_xz_f)**2 + np.abs(tau_yz_f)**2))
    trace_SS = (sigma_x_f**2 + sigma_y_f**2 + sigma_z_f**2
                + 2 * (tau_xy_f**2 + tau_xz_f**2 + tau_yz_f**2))

    I_2 = 0.5 * (I_1_squared - trace_S_conjS)

    I_2_underscore = 0.5 * (I_1_squared_underscore - trace_SS)

    sigma_eq_ka = I_1_squared - k_a**2 * I_2

    sigma_eq_ka_underscore = I_1_squared_underscore - k_a**2 * I_2_underscore

    phi_I1 = np.angle(I_1_underscore)

    phi_eq_ka = np.angle(np.sqrt(sigma_eq_ka_underscore.astype(complex)))

    # amplitude
    sigma_coin = np.sqrt((k_phi * sigma_eq_ka + (1 - k_phi) * abs(sigma_eq_ka_underscore)).astype(complex))

    #phase angle: of phi_eq_ka and phi_eq_ka + pi, take the one closer to phi_I1 (Schmidt & Kraft 2023, Eq. 36)
    d0 = np.abs(np.angle(np.exp(1j * (phi_I1 - phi_eq_ka))))
    d1 = np.abs(np.angle(np.exp(1j * (phi_I1 - phi_eq_ka - np.pi))))
    phi_coin = np.where(d0 < d1, phi_eq_ka, phi_eq_ka + np.pi)

    return sigma_coin * np.exp(1j * phi_coin)

def _EVMS_out_of_phase(self, s):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.
    """
    if s.shape[-2:] != (3, 3):
        raise ValueError('EVMS_out_of_phase only works for biaxial stresses: PSD matrix (f,3,3)')

    sigma_xx_squared = np.real(s[...,0,0])
    sigma_yy_squared = np.real(s[...,1,1])
    tau_xy_squared = np.real(s[...,2,2])
    sigma_xx_sigma_yy = np.sqrt(sigma_xx_squared * sigma_yy_squared)

    phi_yy_phi_xx = np.angle(s[...,0,1])
    phi_xx_phi_yy = np.angle(s[...,1,0])
    phi_xy_phi_xx = np.angle(s[...,0,2])

    s_eq = 0.5  * (sigma_xx_squared + sigma_yy_squared - sigma_xx_sigma_yy * np.cos(phi_xx_phi_yy) + 3 * tau_xy_squared) + 0.5 * np.abs(sigma_xx_squared + sigma_yy_squared * np.exp(2j * phi_yy_phi_xx) - sigma_xx_sigma_yy * np.exp(1j * phi_yy_phi_xx) + 3*tau_xy_squared * np.exp(2j * phi_xy_phi_xx))

    return s_eq

def _Nieslony(self, s, s_af, tau_af,coefficient_load_type):
    """
    Internal function for calculating equivalent stress at one node or at all nodes.
    """
    s = cplane._plane_stress_to_voigt_3d(s)

    G_p = 1/9*_thermoelastic(self,s)
    G_EVMS = _EVMS(self,s)

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
    else:
        raise ValueError('Invalid coefficient_load_type. Please choose either "tension" or "torsion"')

    s_eq = np.real(A**2 * G_p + B**2 * G_EVMS + 2 * A * B * rp_EMS * np.sqrt(np.maximum(G_p * G_EVMS, 0)))

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
