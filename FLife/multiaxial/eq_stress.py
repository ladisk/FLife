import numpy as np
from . import cplane
from ..spectralData import SpectralData
from scipy.optimize import minimize

class EqStress(SpectralData):
    

    def __init__(self, input=None, window='hann', nperseg=1280,
                 noverlap=None, psd_trim_length=None, 
                 T=None, fs=None, rg=None,
                 **kwargs):
        
        # Class in instantiated with EqStress(input), input is tuple (PSD,freq)
        if isinstance(input, tuple):
            SpectralData.__init__(self,input=input)
            if T is not None and fs is not None:
                self.t = T
                self.fs = fs
        # Class instance is instantiated with spectralData(input) and inherited with EqStress(spectral_data). Input is an instance of SpectralData class
        elif isinstance(input, SpectralData):
            self.spectral_data = input
            self.multiaxial_psd = self.spectral_data.multiaxial_psd
            if hasattr(input,'t') and hasattr(input,'fs'):
                self.t = input.t
                self.fs = input.fs
        else:
            raise Exception('Unrecognized Input Error. `input` should be tuple with 2 elements (PSD matrix, freq vector).')


    def is_multiaxial_psd_4d(self):
        '''Check if the multiaxial PSD is 4D (i.e., multiple points on the model)'''
        return len(self.multiaxial_psd[0].shape) == 4


    def loop_over_points(self, criterion, *args, **kwargs):
        '''Loop the selected criterion over multiple points on the model'''    
        s_eq_multi_point = np.empty((self.multiaxial_psd[0].shape[:2]))
        print(s_eq_multi_point.shape)
        for i in range(len(s_eq_multi_point)):
            s_eq = criterion(s=self.multiaxial_psd[0][i],*args, **kwargs)
            s_eq_multi_point[i] = s_eq
        self.eq_psd_multipoint = (s_eq_multi_point, self.multiaxial_psd[1]) #(multipoint psd, freq)


    def maxnormal(self,search_method='local'):
        '''
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum normal stress.
        
        --------
        
        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d(): 
            self.loop_over_points(self._maxnormal,search_method = 'local')
        
        # psd in single-point
        else: 
            s_eq = self._maxnormal(self.multiaxial_psd[0],search_method=search_method)
            self.set_eq_stress(s_eq, self.multiaxial_psd[1])


    def maxshear(self,search_method='local'):
        '''
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum shear stress.
        
        --------
            
        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d():
            self.loop_over_points(self._maxshear,search_method = 'local')
        
        # psd is single-point
        else:
            s_eq = self._maxshear(self.multiaxial_psd[0],search_method=search_method)
            self.set_eq_stress(s_eq, self.multiaxial_psd[1])

    def maxnormalandshear(self, s_af, tau_af, search_method='local'):
        '''
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum normal and shear stress.
        
        Critical plane is based on max variance of shear stress

        --------
            
        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d():
            self.loop_over_points(self._maxnormalandshear, search_method='local', s_af=s_af, tau_af=tau_af)
        
        # psd is single-point
        else:
            s_eq = self._maxnormalandshear(self.multiaxial_psd[0], s_af=s_af, tau_af=tau_af, search_method=search_method)
            self.set_eq_stress(s_eq, self.multiaxial_psd[1])
    
    def EVMS(self):
        '''
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using ther equivalent von Misses stress in frequency domain (EVMS)
    
        --------
        -Preumont, A., & Piéfort, V. (1994);
        Predicting Random High-Cycle Fatigue Life With Finite Elements.
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d():
            self.loop_over_points(self._EVMS)
        
        # psd is single-point
        else:
            s_eq = self._EVMS(self.multiaxial_psd[0])
            self.set_eq_stress(s_eq, self.multiaxial_psd[1])

    def cs(self, s_af, tau_af):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using the C-S criterion.
        
        --------

        -Carpinteri A, Spagnoli A and Vantadori S; 
        Reformulation in the frequency domain of a critical plane-based multiaxial fatigue criterion,
        Int J Fat, 2014
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d():
            self.loop_over_points(self._cs, s_af=s_af, tau_af=tau_af)
        
        # psd is single-point
        else:
            s_eq = self._cs(self.multiaxial_psd[0], s_af=s_af, tau_af=tau_af)
            self.set_eq_stress(s_eq, self.multiaxial_psd[1])

    def multiaxial_rainflow(self):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, for use in frequency domain multiaxial rainflow.

        ONLY WORKS FOR BIAXIAL STRESSES: PSD MATRIX (f,3,3)
    
        --------

        -Pitoiset, Xavier, and André Preumont; 
        Spectral methods for multiaxial random fatigue analysis of metallic structures, 
        International journal of fatigue, 2000
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d():
            self.loop_over_points(self._multiaxial_rainflow)
        
        # psd is single-point
        else:
            s_eq = self._multiaxial_rainflow(self.multiaxial_psd[0])
            self.set_eq_stress(s_eq, self.multiaxial_psd[1])

    def TE(self):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using the thermoelasticity based criterion, 
        simulating the equivalent stress, detected with the thermal camera

        --------

        -Šonc J, Zaletelj K and Slavič J;
        Application of thermoelasticity in the frequency-domain multiaxial vibration-fatigue criterion
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d():
            self.loop_over_points(self._TE)
        
        # psd is single-point
        else:
            s_eq = self._TE(self.multiaxial_psd[0])
            self.set_eq_stress(s_eq, self.multiaxial_psd[1])


    # def maxnormal_old(self, search_method='local'):
    #     '''Converts the stress tensor at one node to equivalent,
    #     scalar psd stress, using method of maximum normal stress.
        
    #     --------

    #     -Nieslony, Adam and Macha, Ewald (2007);
    #     Spectral method in multiaxial random fatigue
    #     '''

    #     s = self.multiaxial_psd[0]
    #     freq = self.multiaxial_psd[1]
    #     df = freq[1] - freq[0]

    #     l1, m1, n1= cplane.maxvariance_old(s,df,method='maxnormal',search_method=search_method)
        
    #     a = np.asarray([l1**2, m1**2, n1**2,
    #                 2*l1*m1, 2*l1*n1, 2*m1*n1])
        
    #     s_eq = np.einsum('i,zij,j', a, s, a)
    #     self.set_eq_stress(s_eq, freq)



    def _maxnormal(self, s, search_method):
        '''
        Internal function for calculating equivalent stress at one node.
        '''
        #s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]
        
        l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.maxvariance(s, df, method='maxnormal',search_method=search_method)
        
        a = np.asarray([l1**2, m1**2, n1**2,
                    2*l1*m1, 2*l1*n1, 2*m1*n1])

        s_eq = np.einsum('i,zij,j', a, s, a)
        return s_eq


    def _maxshear(self, s, search_method):
        '''
        Internal function for calculating equivalent stress at one node.
        '''
        #s = self.multiaxial_psd[0]
        freq = self.multiaxial_psd[1]
        df = freq[1] - freq[0]

        l1, m1, n1, l2, m2, n2, l3, m3, n3 = cplane.maxvariance(s, df, method='maxshear', search_method=search_method)

        a = np.asarray([l1**2-l3**2, m1**2-m3**2, n1**2-n3**2,
                2*(l1*m1-l3*m3), 2*(l1*n1-l3*n3), 2*(m1*n1-m3*n3)])

        s_eq = np.einsum('i,zij,j', a, s, a)
        return s_eq


    def _maxnormalandshear(self, s, s_af, tau_af, search_method='local'):
        '''
        Internal function for calculating equivalent stress at one node.
        '''
        #s = self.multiaxial_psd[0]
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
        return s_eq


    def _EVMS(self, s):
        '''
        Internal function for calculating equivalent stress at one node.
        '''
        #s = self.multiaxial_psd[0]
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
        #s = self.multiaxial_psd[0]
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
        return s_eq


    def _multiaxial_rainflow(self, s):
        '''
        Internal function for calculating equivalent stress at one node.
        '''
        if s.shape[1] == 6 and s.shape[2] == 6:
            raise ValueError('Multiaxial rainflow only works for biaxial stresses: PSD matrix (f,3,3)')
        #s = self.multiaxial_psd[0]
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
            
            result = minimize(Psi_m, initial_guess, constraints=[cons], method='trust-constr')


            if not result.success:
                print("Optimization failed:", result.message)

                s_eq[i] -Psi_m(initial_guess)  
            # Extract the optimized c values
            else:    
                c_opt = result.x

                s_eq[i] = -Psi_m(c_opt)
        return s_eq


    def _TE(self,s):
        '''
        Internal function for calculating equivalent stress at one node.
        '''
        #s = self.multiaxial_psd[0]
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


    def set_eq_stress(self,eq_psd,f):

        self.psd = np.column_stack((f, eq_psd))
        # needed parameters for time-history generation
        if hasattr(self,'t') and hasattr(self, 'fs'):
            self._set_time_history(f=f, psd=eq_psd, T=self.t, fs=self.fs)
        if hasattr(self,'psd'):
            self.PSD_splitting = ('equalAreaBands', 1) 
            self._calculate_coefficients()