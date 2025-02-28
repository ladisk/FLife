import numpy as np
from . import cplane
from . import criteria
from ..spectralData import SpectralData
from scipy.optimize import minimize

class EquivalentStress(SpectralData): #equivalentstress
    """EquivalentStress object contains multiaxial data, based on input power spectral density (PSD).
    It is used to convert multiaxial stress states into equivalent uniaxiail stress states, that can be used in fatigue life estimation.

    EquivalentStress inherits from SpectralData class, so all methods from SpectralData are available for fatigue analysis.
    If instance of SpectralData is passed as input, the multiaxial PSD is inherited, otherwise it is created from input PSD.
    
    -----
    
    The following multiaxial criteria are available for equivalent stress calculation:
    - max_normal: Maximum normal stress on critical plane
    - max_shear: Maximum shear stress on critical plane
    - max_normal_and_shear: Maximum normal and shear stress on critical plane
    - EVMS: Equivalent von Misses stress
    - cs: Carpinteri-Spagnoli criterion
    - multiaxial_rainflow: Frequency-based multiaxial rainflow criterion
    - thermoelastic: Thermoelasticity-based criterion
    - EVMS_out_of_phase: EVMS adaptation for out-of-phase components
    - Nieslony: Nieslony criterion, combining von Mises and hydrostatic stresses
    - Lemaitre: Equivalent Lemaitre stress
    - liwi: LiWI approach
    - coin_liwi: COIN-LiWI method

    """

    def __init__(self, input=None, window='hann', nperseg=1280,
                 noverlap=None, psd_trim_length=None, 
                 T=None, fs=None, rg=None,
                 **kwargs):
        """
        Class constructor for EquivalentStress object.

        :param input: dictionary or instance of SpectralData class.
                Input data for EquivalentStress object. If dictionary, it should contain keys: `PSD` and `f`
                or keys 'amplitude_spectrum' and 'f'.
                PSD should be an array with shape (f,6,6) for 3D stress state or (f,3,3) for 2D stress state.
                If multi-point PSD is provided, the shape should be (N,f,6,6) or (N,f,3,3) respectively.
                Amplitude spectrum should be an array with shape (f,3) for 2D stress state or (f,6) for 3D stress state.
                If multi-point amplitude spectrum is provided, the shape should be (N,f,3) or (N,f,6) respectively.
                If instance of SpectralData class is passed, the multiaxial PSD is inherited.
        :param window: str or tuple or array_like, optional
                Desired window to use. Defaults to ‘hann’.
        :param nperseg:  int, optional
                Length of each segment. Defaults to 1280.
        :param noverlap: int, optional
                Number of points to overlap between segments. If None, 
                noverlap = nperseg / 2. Defaults to None.
        :param psd_trim_length: int, optional
                Number of frequency points to be used for PSD.
                Defaults to None.
        :param T: int, float, optional
                Length of time history when random process is defined by `input` parameter 'GUI'
                or by PSD and frequency vector. If T and fs are provided, time histoy is generated.
                Defaults to None.
        :param fs: int, float, optional
                Sampling frequency of time history when random process is defined by `input` parameter 
                'GUI' or by PSD and frequency vector. If T and fs are provided, time histoy is generated.
                Defaults to None.
        :param rg: numpy.random._generator.Generator, optional
                Random generator controls phase of generated time history, when `input` is 'GUI' or 
                (PSD, frequency vector).
                Defaults to None.

        """
        # Class in instantiated with EquivalentStress(input), input is dictionary or tuple (PSD,freq)
        if isinstance(input, dict) or isinstance(input, tuple):
            SpectralData.__init__(self,input=input)
            if T is not None and fs is not None:
                self.t = T
                self.fs = fs
        # Class instance is instantiated with SpectralData(input) and inherited with EquivalentStress(spectral_data). Input is an instance of SpectralData class
        elif isinstance(input, SpectralData):
            self.spectral_data = input
            if hasattr(input,'multiaxial_psd'):
                self.multiaxial_psd = self.spectral_data.multiaxial_psd
            elif hasattr(input,'multiaxial_amplitude_spectrum'):
                self.multiaxial_amplitude_spectrum = self.spectral_data.multiaxial_amplitude_spectrum
            if hasattr(input,'t') and hasattr(input,'fs'):
                self.t = input.t
                self.fs = input.fs
            self.multipoint = self.spectral_data.multipoint
        else:
            raise Exception('Unrecognized Input Error. `input` should be a dictionary with keys: `PSD` and `f`).')


    # def is_multiaxial_psd_4d(self):
    #     """Check if the multiaxial PSD is 4D (i.e., multiple points on the model)"""
    #     return len(self.multiaxial_psd[0].shape) == 4


    def loop_over_points(self, criterion, *args, **kwargs):
        """Loop the selected criterion over multiple points on the model"""    
        if hasattr(self,'multiaxial_psd'):
            s_eq_multi_point = np.empty((self.multiaxial_psd[0].shape[:2]))
        elif hasattr(self,'multiaxial_amplitude_spectrum'):
            s_eq_multi_point = np.empty((self.multiaxial_amplitude_spectrum[0].shape[:2]),dtype=complex)
        
        for i in range(len(s_eq_multi_point)):
            if hasattr(self,'multiaxial_psd'):
                s_eq = criterion(self,s=self.multiaxial_psd[0][i],*args, **kwargs)
            elif hasattr(self,'multiaxial_amplitude_spectrum'):
                s_eq = criterion(self,s=self.multiaxial_amplitude_spectrum[0][i],*args, **kwargs)
            s_eq_multi_point[i] = s_eq
        if hasattr(self,'multiaxial_psd'):
            self.eq_psd_multipoint = (s_eq_multi_point, self.multiaxial_psd[1]) #(multipoint psd, freq)
        elif hasattr(self,'multiaxial_amplitude_spectrum'):
            #calculate psd from amplitude spectrum at each point at each frequency
            self.eq_psd_multipoint = (abs(s_eq_multi_point**2), self.multiaxial_amplitude_spectrum[1])

    def set_eq_stress(self,eq_psd,f):
        """Set equivalent stress to the object. Also generate im ehistory if T and fs are provided"""
        self.psd = np.column_stack((f, eq_psd))
        # needed parameters for time-history generation
        if hasattr(self,'t') and hasattr(self, 'fs'):
            self._set_time_history(f=f, psd=eq_psd, T=self.t, fs=self.fs)
        if hasattr(self,'psd'):
            self.PSD_splitting = ('equalAreaBands', 1) 
            self._calculate_coefficients()

    def select_critical_point(self, point_index):
        """Manually select a critical point from multipoint equivalent stress
        
        :param point_index: int
                Index of the critical point

        """
        self.set_eq_stress(eq_psd=self.eq_psd_multipoint[0][point_index], f=self.eq_psd_multipoint[1])

    def max_normal(self,search_method='local'): #max_normal
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum normal stress.
        
        :param search_method: str, optional, default 'local'
                Search method for optimization. Options are 'local' or 'global'. Local is prefered, unless the optimization fails.
        
        --------

        - Nieslony, Adam and Macha, Ewald (2007);
            Spectral method in multiaxial random fatigue

        """
        # psd is multiple-point
        if self.multipoint: 
            self.loop_over_points(criterion=criteria._max_normal,search_method = 'local')
        
        # psd in single-point
        else: 
            s_eq = criteria._max_normal(self,s=self.multiaxial_psd[0],search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])


    def max_shear(self,search_method='local'):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum shear stress.
        
        :param search_method: str, optional, default 'local'
                Search method for optimization. Options are 'local' or 'global'. Local is prefered, unless the optimization fails.
        
        --------
            
        - Nieslony, Adam and Macha, Ewald (2007);
            Spectral method in multiaxial random fatigue

        """
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._max_shear,search_method = 'local')
        
        # psd is single-point
        else:
            s_eq = criteria._max_shear(self,s=self.multiaxial_psd[0],search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

    def max_normal_and_shear(self, s_af, tau_af, search_method='local'):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum normal and shear stress.
        
        Critical plane is based on max variance of shear stress
        
        :param s_af: float
            Fully reversed torsion-fatigue limit. Used for calculating material fatigue coefficient K [1].
        :param tau_af: float
            Fully reversed torsion-fatigue limit. Used Used for calculating material fatigue coefficient K [1].
        
        --------
            
        - Nieslony, Adam and Macha, Ewald (2007);
            Spectral method in multiaxial random fatigue

        - Matjaz Mrsnik, Janko Slavic and Miha Boltezar
            Multiaxial Vibration Fatigue A Theoretical and Experimental Comparison.
            Mechanical Systems and Signal Processing, 2016

        """
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._max_normal_and_shear, search_method='local', s_af=s_af, tau_af=tau_af)
        
        # psd is single-point
        else:
            s_eq = criteria._max_normal_and_shear(self,s=self.multiaxial_psd[0], s_af=s_af, tau_af=tau_af, search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])
    
    def EVMS(self):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using ther equivalent von Misses stress in frequency domain (EVMS)
    
        --------

        - Preumont, A., & Piéfort, V. (1994);
            Predicting Random High-Cycle Fatigue Life With Finite Elements.

        """
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._EVMS)
        
        # psd is single-point
        else:
            s_eq = criteria._EVMS(self,s=self.multiaxial_psd[0])
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

    def cs(self, s_af, tau_af):
        """Converts the stress tensor at one node to equivalent,
        scalar psd stress, using the C-S criterion.
        
        :param s_af: float
            Fully reversed torsion-fatigue limit.
        :param tau_af: float
            Fully reversed torsion-fatigue limit.
        
        --------

        - Carpinteri A, Spagnoli A and Vantadori S; 
            Reformulation in the frequency domain of a critical plane-based multiaxial fatigue criterion,
            Int J Fat, 2014

        """
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._cs, s_af=s_af, tau_af=tau_af)
        
        # psd is single-point
        else:
            s_eq = criteria._cs(self,s=self.multiaxial_psd[0], s_af=s_af, tau_af=tau_af)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

    def multiaxial_rainflow(self):
        """Converts the stress tensor at one node to equivalent,
        scalar psd stress, for use in frequency domain multiaxial rainflow.

        ONLY WORKS FOR BIAXIAL STRESSES: PSD MATRIX (f,3,3) - single point or (N,f,3,3) - multiple points
    
        --------

        - Pitoiset, Xavier, and André Preumont; 
            Spectral methods for multiaxial random fatigue analysis of metallic structures, 
            International journal of fatigue, 2000

        """
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._multiaxial_rainflow)
        
        # psd is single-point
        else:
            s_eq = criteria._multiaxial_rainflow(self,s=self.multiaxial_psd[0])
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

    def thermoelastic(self):
        """Converts the stress tensor at one node to equivalent,
        scalar psd stress, using the thermoelasticity based criterion, 
        simulating the equivalent stress, detected with the thermal camera

        --------

        - Šonc J, Zaletelj K and Slavič J;
            Application of thermoelasticity in the frequency-domain multiaxial vibration-fatigue criterion, 
            Mechanical Systems and Signal Processsing, 2025

        """
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._thermoelastic)
        
        # psd is single-point
        else:
            s_eq = criteria._thermoelastic(self,s=self.multiaxial_psd[0])
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

    def liwi(self):
        """Converts the stress tensor at one node to equivalent
        stress, using the LiWI approach.

        ONLY WORKS WITH BIAXIAL AMPLITUDE SPECTRUM (f,3) - single point or (N,f,3) - multiple points

        --------

        - Alexander T. Schmidt, Nimish Pandiya,
            Extension of the static equivalent stress hypotheses to linearly vibrating systems using wave interference – The LiWI approach,
            International Journal of Fatigue, 2021,

        """
        if not hasattr(self,'multiaxial_amplitude_spectrum'):
            raise Exception('Input Error. LiWI approach only works with multiaxial amplitude spectrum.')
        
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._liwi)
        
        # psd is single-point
        else:
            s_eq_amplitude_spectrum = criteria._liwi(self,s=self.multiaxial_amplitude_spectrum[0])

            for i in range(len(s_eq_amplitude_spectrum)):
                s_eq_psd = np.dot(s_eq_amplitude_spectrum[i], np.conj(s_eq_amplitude_spectrum[i].T))

            self.set_eq_stress(eq_psd=s_eq_psd, f=self.multiaxial_psd[1])


    def coin_liwi(self, k_a, k_phi):
        """Converts the stress tensor at one node to equivalent
        stress, using the COIN-LiWI method.

        ONLY WORKS WITH 3D AMPLITUDE SPECTRUM (f,6) - single point or (N,f,6) - multiple points
        
        :param k_a: float
            Tension shear strength ratio. (from article: 1.70 for aluminum alloy, 1.64 for structural steel, 1.43 for cast iron)
        :param k_phi: float
            Phase influence factor (from article: 0.90 for aluminum alloy, 0.85 for structural steel, 1.10 for cast iron)
        
        --------

        - Alexander T. Schmidt, Jan Kraft,
            A new equivalent stress approach based on complex invariants: The COIN LiWI method,
            International Journal of Fatigue, 2023
        
        """
        if not hasattr(self,'multiaxial_amplitude_spectrum'):
            raise Exception('Input Error. COIN LiWI method only works with multiaxial amplitude spectrum.')
        
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._coin_liwi, k_a=k_a, k_phi=k_phi)
        
        # psd is single-point
        else:
            s_eq_amplitude_spectrum = criteria._coin_liwi(self,s=self.multiaxial_amplitude_spectrum[0], k_a=k_a, k_phi=k_phi)

            for i in range(len(s_eq_amplitude_spectrum)):
                s_eq_psd = np.dot(s_eq_amplitude_spectrum[i], np.conj(s_eq_amplitude_spectrum[i].T))

            self.set_eq_stress(eq_psd=s_eq_psd, f=self.multiaxial_psd[1])


    def EVMS_out_of_phase(self):
        """Converts the stress tensor at one node to equivalent
        stress, using the phase-sensitive EVMS criterion, adjusted for out-of-phase components.

        ONLY WORKS WITH BIAXIAL PSD (f,3,3) - single point or (N,f,3,3) - multiple points
               
        --------

        - Bonte, Martijn HA and de Boer, Andries and Liebregts R
            Determining the von Mises stress power spectral density for frequency domain fatigue analysis including out-of-phase stress components,
            Journal of Sound and Vibration, 2007
        
        """
        
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._EVMS_out_of_phase)
        
        # psd is single-point
        else:
            s_eq_psd = criteria._EVMS_out_of_phase(self,s=self.multiaxial_psd[0])

            self.set_eq_stress(eq_psd=s_eq_psd, f=self.multiaxial_psd[1])


    def Nieslony(self, s_af, tau_af, coefficient_load_type='tension'):
        """Converts the stress tensor at one node to equivalent,
        scalar psd stress, using the Nieaslony criterion, combining von Mises and hydrostatic stresses.

        :param s_af: float
            Fully reversed torsion-fatigue limit.
        :param tau_af: float
            Fully reversed torsion-fatigue limit.
        :param coefficient_load_type: str
            Type of load with which the material fatigue properties were determined. Options are 'tension' for tension-compression load or 'torsion'.
        
        --------

        - Adam Niesłony, Michał Böhm, Robert Owsiński, Artur Dziura, Karol Czekaj,
            Integrating von Mises and hydrostatic stresses in frequency domain multiaxial fatigue criteria for vibration fatigue analysis,
            Mechanical Systems and Signal Processing, 2025

        """
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._Nieslony, s_af=s_af, tau_af=tau_af, coefficient_load_type=coefficient_load_type)
        
        # psd is single-point
        else:
            s_eq = criteria._Nieslony(self,s=self.multiaxial_psd[0], s_af=s_af, tau_af=tau_af, coefficient_load_type=coefficient_load_type)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])

    def Lemaitre(self, poisson_ratio):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using ther equivalent Lemaitre stress in frequency domain
    
        --------

        - Jingran Ge, Yi Sun, Song Zhou;
            Fatigue life estimation under multiaxial random loading by means of the equivalent Lemaitre stress and multiaxial S–N curve methods.
            International Journal of Fatigue, 2015
            
        """
        # psd is multiple-point
        if self.multipoint:
            self.loop_over_points(criterion=criteria._Lemaitre, poisson_ratio=poisson_ratio)
        
        # psd is single-point
        else:
            s_eq = criteria._Lemaitre(self,s=self.multiaxial_psd[0], poisson_ratio=poisson_ratio)
            self.set_eq_stress(eq_psd=s_eq, f=self.multiaxial_psd[1])