import numpy as np
from . import cplane
from scipy.optimize import minimize

def _max_normal(self, s, search_method):
    """
    Internal function for calculating equivalent stress at one node.
    """
    
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
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    K = 2 * tau_af / s_af - 1
    l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.max_variance(multiaxial_psd=s, df=df, method='maxshear',K=K, search_method=search_method)
    
    a = np.asarray([l1**2-l3**2+K*(l1+l3)**2,
                    m1**2-m3**2+K*(m1+m3)**2,
                    n1**2-n3**2+K*(n1+n3)**2,
                    2*(l1*m1-l3*m3+K*(l1+l3)*(m1+m3)),
                    2*(l1*n1-l3*n3+K*(l1+l3)*(n1+n3)),
                    2*(m1*n1-m3*n3+K*(m1+m3)*(n1+n3))]) / (1 + K) 

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


def _cs(self, s, s_af, tau_af):
    """
    Internal function for calculating equivalent stress at one node.
    """
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    # .. Find critical plane.
    C = cplane.csrandom(multiaxial_psd=s, df=df, s_af=s_af, tau_af=tau_af)
    
    # .. Transform stress.
    s = np.einsum('ij,zjk,lk->zil', C, s, C)
    
    # .. Calculate equivalent PSD.
    sww = s[:, 2, 2]
    suw = s[:, -1, -1,]

    s_eq = sww.real + suw.real * (s_af/tau_af)
    return s_eq


def _multiaxial_rainflow(self, s):
    """
    Internal function for calculating equivalent stress at one node.
    """
    if s.shape[1] == 6 and s.shape[2] == 6:
        raise ValueError('Multiaxial rainflow only works for biaxial stresses: PSD matrix (f,3,3)')
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    

    def constraint(c):
        return np.sum(c**2) - 1

    initial_guess = np.array([0.34,0.33,0.33])

    # Define the constraint for the optimizaton
    cons = {'type': 'eq', 'fun': constraint}

    # Perform the optimization
    s_eq = np.empty((len(s)))
    for i in range(len(s)):
        def Psi_m(c):
            Q = np.outer(c, c)
            return -np.trace(Q@s[i])
        
        result = minimize(fun=Psi_m, x0=initial_guess, constraints=[cons], method='trust-constr')


        if not result.success:
            print("Optimization failed:", result.message)

            s_eq[i] -Psi_m(initial_guess)  
        # Extract the optimized c values
        else:    
            c_opt = result.x

            s_eq[i] = -Psi_m(c_opt)
    return s_eq


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
    """
    s_eq = np.empty((len(s)))

    for i in range(len(s)):
        sigma_xx_squared = s[i,0,0]
        sigma_yy_squared = s[i,1,1]
        tau_xy_squared = s[i,2,2]
        sigma_xx_sigma_yy = np.sqrt(sigma_xx_squared * sigma_yy_squared)

        phi_yy_phi_xx = np.angle(s[i,0,1])
        phi_xx_phi_yy = np.angle(s[i,1,0])
        phi_xy_phi_xx = np.angle(s[i,0,2])

        s_eq[i] = 0.5  * (sigma_xx_squared + sigma_yy_squared - sigma_xx_sigma_yy * np.cos(phi_xx_phi_yy) + 3 * tau_xy_squared) + 0.5 * np.abs(sigma_xx_squared + sigma_yy_squared * np.exp(2j * phi_yy_phi_xx) - sigma_xx_sigma_yy * np.exp(1j * phi_yy_phi_xx) + 3*tau_xy_squared * np.exp(2j * phi_xy_phi_xx))

    return s_eq

def _Nieslony(self, s, s_af, tau_af,coefficient_load_type):
    """
    Internal function for calculating equivalent stress at one node.
    """

    G_p = 1/9*_thermoelastic(self,s)
    G_EVMS = _EVMS(self,s)

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
        
        s_eq[i] = A**2 * G_p[i] + B**2 * G_EVMS[i] + 2 * A * B * rp_EMS * np.sqrt(G_p[i] * G_EVMS[i])
    
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