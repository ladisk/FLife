import numpy as np
from . import cplane
from scipy.optimize import minimize

def _max_normal(self, s, search_method):
    '''
    Internal function for calculating equivalent stress at one node.
    '''
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]
    
    l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.max_variance(multiaxial_psd=s, df=df, method='maxnormal',search_method=search_method)
    
    a = np.asarray([l1**2, m1**2, n1**2,
                2*l1*m1, 2*l1*n1, 2*m1*n1])

    s_eq = np.einsum('i,zij,j', a, s, a)
    return s_eq


def _max_shear(self, s, search_method):
    '''
    Internal function for calculating equivalent stress at one node.
    '''
    
    freq = self.multiaxial_psd[1]
    df = freq[1] - freq[0]

    l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.max_variance(multiaxial_psd=s, df=df, method='maxshear', search_method=search_method)

    a = np.asarray([l1**2-l3**2, m1**2-m3**2, n1**2-n3**2,
            2*(l1*m1-l3*m3), 2*(l1*n1-l3*n3), 2*(m1*n1-m3*n3)])

    s_eq = np.einsum('i,zij,j', a, s, a)
    return s_eq


def _max_normal_and_shear(self, s, s_af, tau_af, search_method='local'):
    '''
    Internal function for calculating equivalent stress at one node.
    '''
    
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
    '''
    Internal function for calculating equivalent stress at one node.
    '''
    
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
    '''
    Internal function for calculating equivalent stress at one node.
    '''
    
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
    '''
    Internal function for calculating equivalent stress at one node.
    '''
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
    '''
    Internal function for calculating equivalent stress at one node.
    '''
    
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


