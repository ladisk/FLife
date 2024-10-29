import numpy as np
from . import cplane
from . import criteria
from ..spectralData import SpectralData
from scipy.optimize import minimize

class EquivalentStress(SpectralData): #equivalentstress
    

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
            s_eq = criterion(self,s=self.multiaxial_psd[0][i],*args, **kwargs)
            s_eq_multi_point[i] = s_eq
        self.eq_psd_multipoint = (s_eq_multi_point, self.multiaxial_psd[1]) #(multipoint psd, freq)

    def set_eq_stress(self,eq_psd,f):

        self.psd = np.column_stack((f, eq_psd))
        # needed parameters for time-history generation
        if hasattr(self,'t') and hasattr(self, 'fs'):
            self._set_time_history(f=f, psd=eq_psd, T=self.t, fs=self.fs)
        if hasattr(self,'psd'):
            self.PSD_splitting = ('equalAreaBands', 1) 
            self._calculate_coefficients()


    def max_normal(self,search_method='local'): #max_normal
        '''
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum normal stress.
        
        --------
        
        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d(): 
            self.loop_over_points(criterion=criteria._max_normal,search_method = 'local')
        
        # psd in single-point
        else: 
            s_eq = criteria._max_normal(self,s=self.multiaxial_psd[0],search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])


    def max_shear(self,search_method='local'):
        '''
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum shear stress.
        
        --------
            
        -Nieslony, Adam and Macha, Ewald (2007);
        Spectral method in multiaxial random fatigue
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d():
            self.loop_over_points(criterion=criteria._max_shear,search_method = 'local')
        
        # psd is single-point
        else:
            s_eq = criteria._max_shear(self,s=self.multiaxial_psd[0],search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

    def max_normal_and_shear(self, s_af, tau_af, search_method='local'):
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
            self.loop_over_points(criterion=criteria._max_normal_and_shear, search_method='local', s_af=s_af, tau_af=tau_af)
        
        # psd is single-point
        else:
            s_eq = criteria._max_normal_and_shear(self,s=self.multiaxial_psd[0], s_af=s_af, tau_af=tau_af, search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])
    
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
            self.loop_over_points(criterion=criteria._EVMS)
        
        # psd is single-point
        else:
            s_eq = criteria._EVMS(self,s=self.multiaxial_psd[0])
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

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
            self.loop_over_points(criterion=criteria._cs, s_af=s_af, tau_af=tau_af)
        
        # psd is single-point
        else:
            s_eq = criteria._cs(self,s=self.multiaxial_psd[0], s_af=s_af, tau_af=tau_af)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

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
            self.loop_over_points(criterion=criteria._multiaxial_rainflow)
        
        # psd is single-point
        else:
            s_eq = criteria._multiaxial_rainflow(self,s=self.multiaxial_psd[0])
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

    def thermoelastic(self):
        '''Converts the stress tensor at one node to equivalent,
        scalar psd stress, using the thermoelasticity based criterion, 
        simulating the equivalent stress, detected with the thermal camera

        --------

        -Šonc J, Zaletelj K and Slavič J;
        Application of thermoelasticity in the frequency-domain multiaxial vibration-fatigue criterion
        '''
        # psd is multiple-point
        if self.is_multiaxial_psd_4d():
            self.loop_over_points(criterion=criteria._thermoelastic)
        
        # psd is single-point
        else:
            s_eq = criteria._thermoelastic(self,s=self.multiaxial_psd[0])
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])


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



