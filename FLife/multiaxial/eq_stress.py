import numpy as np
from . import cplane
from ..spectralData import SpectralData
from scipy.optimize import minimize

class EqStress(SpectralData):
    

    def __init__(self, input=None, window='hann', nperseg=1280,
                 noverlap=None, psd_trim_length=None, 
                 T=None, fs=None, rg=None,
                 **kwargs):
        
        # Class in instantiated with eqStress(input), input is tuple (PSD,freq)
        if isinstance(input, tuple):
            SpectralData.__init__(self,input=input)
            if T is not None and fs is not None:
                self.t = T
                self.fs = fs
        # Class instance is instantiated with spectralData(input) and inherited with eqStress(spectral_data)
        elif isinstance(input, SpectralData):
            self.spectral_data = input
            self.multiaxial_psd = self.spectral_data.multiaxial_psd
            if hasattr(input,'t') and hasattr(input,'fs'):
                self.t = input.t
                self.fs = input.fs
        else:
            raise Exception('Unrecognized Input Error. `input` should be tuple with 2 elements (PSD matrix, freq vector).')



    def maxnormal(self, search_method='local'):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using method of maximum normal stress.
        
        --------
        
        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''
        s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]
        
        l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.maxvariance(s, df, method='maxnormal',search_method=search_method)
        
        a = np.asarray([l1**2, m1**2, n1**2,
                    2*l1*m1, 2*l1*n1, 2*m1*n1])

        s_eq = np.einsum('i,zij,j', a, s, a)
        self.set_eq_stress(s_eq, freq)


    def maxnormal_old(self, search_method='local'):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using method of maximum normal stress.
        
        --------

        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''

        s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]

        l1, m1, n1= cplane.maxvariance_old(s,df,method='maxnormal',search_method=search_method)
        
        a = np.asarray([l1**2, m1**2, n1**2,
                    2*l1*m1, 2*l1*n1, 2*m1*n1])
        
        s_eq = np.einsum('i,zij,j', a, s, a)
        self.set_eq_stress(s_eq, freq)


    def maxshear(self, search_method='local'):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using method of maximum shear stress.
        
        --------
        
        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''
        s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]

        l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.maxvariance(s, df, method='maxshear', search_method=search_method)

        a = np.asarray([l1**2-l3**2, m1**2-m3**2, n1**2-n3**2,
                2*(l1*m1-l3*m3), 2*(l1*n1-l3*n3), 2*(m1*n1-m3*n3)])

        s_eq = np.einsum('i,zij,j', a, s, a)
        self.set_eq_stress(s_eq, freq)


    def maxnormalandshear(self, s_af, tau_af, search_method='local'):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using method of maximum normal stress
        and shear.
        
        Critical plane is based on max variance of shear stress
        
        --------

        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''
        s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]

        K = 2 * tau_af / s_af - 1
        l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.maxvariance(s, df, method='maxshear', search_method=search_method)
        
        a = np.asarray([l1**2-l3**2+K*(l1+l3)**2,
                        m1**2-m3**2+K*(m1+m3)**2,
                        n1**2-n3**2+K*(n1+n3)**2,
                        2*(l1*m1-l3*m3+K*(l1+l3)*(m1+m3)),
                        2*(l1*n1-l3*n3+K*(l1+l3)*(n1+n3)),
                        2*(m1*n1-m3*n3+K*(m1+m3)*(n1+n3))]) / (1 + K)

        s_eq = np.einsum('i,zij,j', a, s, a)
        self.set_eq_stress(s_eq, freq)


    def EVMS(self):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using ther equivalent von Misses stress in frequency domain (EVMS)
    
        --------
        -Preumont, A., & Piéfort, V. (1994);
        Predicting Random High-Cycle Fatigue Life With Finite Elements.
        '''
        s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]

        Q = np.array([
            [1,-1/2,-1/2,0,0,0],
            [-1/2,1,-1/2,0,0,0],
            [-1/2,-1/2,1,0,0,0],
            [0,0,0,3,0,0],
            [0,0,0,0,3,0],
            [0,0,0,0,0,3]])

        s_eq = np.trace(np.einsum('ij,kjl->kil', Q, s),axis1=1,axis2=2)
        self.set_eq_stress(s_eq, freq)

    def cs(self, s_af, tau_af):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using the C-S criterion.
        
        --------

        -Carpinteri A, Spagnoli A and Vantadori S, Reformulation in the frequency domain of a critical
        plane-based multiaxial fatigue criterion, Int J Fat, 2014
        '''
        s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]

        # .. Find critical plane.
        C = cplane.csrandom(s, df, s_af, tau_af)
        
        # .. Transform stress.
        s = np.einsum('ij,zjk,lk->zil', C, s, C)
        
        # .. Calculate equivalent PSD.
        sww = s[:, 2, 2]
        suw = s[:, -1, -1,]

        s_eq = sww.real + suw.real * (s_af/tau_af)
        self.set_eq_stress(s_eq, freq)

    def multiaxial_rainflow(self):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, for use in frequency domain multiaxial rainflow.

        ONLY WORKS WITH BIAXIAL STRESSES: PSD MATRIX (f,3,3)
    
        --------

        -Pitoiset, Xavier, and André Preumont. 
        "Spectral methods for multiaxial random fatigue analysis of metallic structures." 
        International journal of fatigue, 2000
        '''

        s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]

        

        def constraint(c):
            return np.sum(c**2) - 1

        initial_guess = np.array([0.34,0.33,0.33])

        # Define the constraint for the optimizaton
        cons = {'type': 'eq', 'fun': constraint}

        # Perform the optimization
        print(len(s))
        s_eq = np.empty((len(s)))
        for i in range(len(s)):
            def Psi_m(c):
                Q = np.outer(c, c)
                return -np.trace(Q@s[i])
            
            result = minimize(Psi_m, initial_guess, constraints=[cons], method='trust-constr')


            if not result.success:
                print("Optimization failed:", result.message)

                s_eq[i] -Psi_m(initial_guess)  
            # Extract the optimized c values
            else:    
                c_opt = result.x

                s_eq[i] = -Psi_m(c_opt)
        self.set_eq_stress(s_eq, freq)



    def TE(self):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using the thermoelasticity based criterion.
    
        --------

        -
        '''
        s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]

        Q = np.array([
            [1,1,1,0,0,0],
            [1,1,1,0,0,0],
            [1,1,1,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]])

        s_eq = np.trace(np.einsum('ij,kjl->kil', Q, s),axis1=1,axis2=2)
        self.set_eq_stress(s_eq, freq)

    def set_eq_stress(self,eq_psd,f):
        self.psd = np.column_stack((f, eq_psd))
        # needed parameters for time-history generation
        if hasattr(self,'t') and hasattr(self, 'fs'):
        #if self.t is not None and self.fs is not None:
            self._set_time_history(f=f, psd=eq_psd, T=self.t, fs=self.fs)