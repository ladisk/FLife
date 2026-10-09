import warnings
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
    A multiaxial amplitude spectrum A is converted to PSD as S_ij = A_i conj(A_j)/df (df from the uniform frequency vector),
    the same A**2/df convention as for the uniaxial amplitude spectrum. LiWI and COIN-LiWI give the equivalent PSD |A_eq|**2/df.
    A is the RMS amplitude per frequency line. For one-sided peak amplitudes (e.g. FFT amplitudes as used by
    LiWI and COIN-LiWI, Schmidt and Kraft 2023) pass A/sqrt(2).
    
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

    For a review of the implemented multiaxial criteria, see:
    Jaša Šonc and Janko Slavič, Review of Multiaxial Vibration Fatigue Spectral
    Methods—With Open-Source Support, Applied Nonlinear Dynamics and Vibrations,
    2026. https://www.sciltp.com/journals/andv/articles/2606004129

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
            SpectralData.__init__(self, input=input, T=T, fs=fs, rg=rg)
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
                self.rg = getattr(input, 'rg', None)
            self.multipoint = self.spectral_data.multipoint
        else:
            raise Exception('Unrecognized Input Error. `input` should be a dictionary with keys: `PSD` and `f`).')


    # def is_multiaxial_psd_4d(self):
    #     """Check if the multiaxial PSD is 4D (i.e., multiple points on the model)"""
    #     return len(self.multiaxial_psd[0].shape) == 4


    def _freq(self):
        """Frequency vector of the multiaxial input."""
        if hasattr(self, 'multiaxial_psd'):
            return self.multiaxial_psd[1]
        if hasattr(self, 'multiaxial_amplitude_spectrum'):
            return self.multiaxial_amplitude_spectrum[1]
        raise Exception('Input Error. Multiaxial PSD or amplitude spectrum is required.')

    def _psd_matrix(self):
        """Multiaxial PSD matrix and frequency vector. An amplitude spectrum A is converted
        with S_ij = A_i conj(A_j) / df."""
        if hasattr(self, 'multiaxial_psd'):
            return self.multiaxial_psd
        f = self._freq()
        a = self.multiaxial_amplitude_spectrum[0]
        return a[..., :, None] * np.conj(a[..., None, :]) / (f[1] - f[0]), f

    def loop_over_points(self, criterion, *args, **kwargs):
        """Deprecated. Loop the criterion over the points of a multipoint input and set `eq_psd_multipoint`."""
        warnings.warn('`loop_over_points` is deprecated; the criteria now process all points at once.',
                      DeprecationWarning, stacklevel=2)
        if criterion in (criteria._liwi, criteria._coin_liwi) and hasattr(self, 'multiaxial_amplitude_spectrum'):
            a, f = self.multiaxial_amplitude_spectrum
            s_eq = np.array([criterion(self, s=ai, *args, **kwargs) for ai in a])
            self.eq_psd_multipoint = (np.abs(s_eq)**2, f)
            return
        psd, f = self._psd_matrix()
        s_eq = np.array([criterion(self, s=si, *args, **kwargs) for si in psd])
        self.eq_psd_multipoint = (np.real(s_eq), f)

    def _all_points(self, criterion, *args, **kwargs):
        """Apply a criterion that accepts all points at once (N, f, ...) and set `eq_psd_multipoint`."""
        psd, f = self._psd_matrix()
        s_eq_multi_point = criterion(self, s=psd, *args, **kwargs)
        self.eq_psd_multipoint = (np.real(s_eq_multi_point), f) #(multipoint psd, freq)

    def _amplitude_criterion(self, criterion, *args, **kwargs):
        """Apply an amplitude spectrum criterion (single or multiple points) and set the equivalent PSD |A|^2/df."""
        a, f = self.multiaxial_amplitude_spectrum
        s_eq_psd = np.abs(criterion(self, s=a, *args, **kwargs))**2 / (f[1] - f[0])
        if self.multipoint:
            self.eq_psd_multipoint = (s_eq_psd, f)
        else:
            self.set_eq_stress(eq_psd=s_eq_psd, f=f)

    def set_eq_stress(self,eq_psd,f):
        """Set equivalent stress to the object. Also generate im ehistory if T and fs are provided"""
        self.psd = np.column_stack((f, eq_psd))
        # needed parameters for time-history generation
        if hasattr(self,'t') and hasattr(self, 'fs'):
            self._set_time_history(f=f, psd=eq_psd, T=self.t, fs=self.fs, rg=getattr(self, 'rg', None))
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
                Critical plane search. 'local': grid over plane normals plus a local
                refinement, for all points at once. 'global': additionally runs
                differential evolution at each point and keeps the better plane (slow).
        
        --------

        - Nieslony, Adam and Macha, Ewald (2007);
            Spectral method in multiaxial random fatigue

        """
        # psd is multiple-point
        if self.multipoint: 
            self._all_points(criterion=criteria._max_normal, search_method=search_method)
        
        # psd in single-point
        else: 
            psd, f = self._psd_matrix()
            s_eq = criteria._max_normal(self,s=psd, search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=f)


    def max_shear(self,search_method='local'):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum shear stress.
        
        :param search_method: str, optional, default 'local'
                Critical plane search. 'local': grid over plane normals plus a local
                refinement, for all points at once. 'global': additionally runs
                differential evolution at each point and keeps the better plane (slow).
        
        --------
            
        - Nieslony, Adam and Macha, Ewald (2007);
            Spectral method in multiaxial random fatigue

        """
        # psd is multiple-point
        if self.multipoint:
            self._all_points(criterion=criteria._max_shear, search_method=search_method)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq = criteria._max_shear(self,s=psd, search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=f)

    def max_normal_and_shear(self, s_af, tau_af, search_method='local'):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum normal and shear stress.
        
        The critical plane maximises the variance of the equivalent stress
        (Bedkowski and Macha). For uniaxial stress this plane is not at 45 deg,
        so the equivalent PSD is ((sqrt(1+K^2) + K) / (1+K))^2 times the input,
        with K = 2*tau_af/s_af - 1.
        
        :param s_af: float
            Fully reversed bending-fatigue limit. Used for calculating material fatigue coefficient K [1].
        :param tau_af: float
            Fully reversed torsion-fatigue limit. Used Used for calculating material fatigue coefficient K [1].
        :param search_method: str, optional, default 'local'
                Critical plane search, see `max_normal`.
        
        --------
            
        - Nieslony, Adam and Macha, Ewald (2007);
            Spectral method in multiaxial random fatigue

        - Matjaz Mrsnik, Janko Slavic and Miha Boltezar
            Multiaxial Vibration Fatigue A Theoretical and Experimental Comparison.
            Mechanical Systems and Signal Processing, 2016

        """
        # psd is multiple-point
        if self.multipoint:
            self._all_points(criterion=criteria._max_normal_and_shear, search_method=search_method, s_af=s_af, tau_af=tau_af)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq = criteria._max_normal_and_shear(self,s=psd, s_af=s_af, tau_af=tau_af, search_method=search_method)
            self.set_eq_stress(eq_psd=s_eq, f=f)
    
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
            self._all_points(criterion=criteria._EVMS)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq = criteria._EVMS(self,s=psd)
            self.set_eq_stress(eq_psd=s_eq, f=f)

    def cs(self, s_af, tau_af, T=None):
        """Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using the C-S criterion.
        
        :param s_af: float
            Fully reversed bending-fatigue limit.
        :param tau_af: float
            Fully reversed torsion-fatigue limit.
        :param T: float, optional
            Time span in s for the expected number of cycles N1 = nu0*T in the
            Davenport peak factor that sets the critical plane. Defaults to the
            T given to EquivalentStress or SpectralData; raises ValueError if neither is set.
        
        --------

        - Carpinteri A, Spagnoli A and Vantadori S; 
            Reformulation in the frequency domain of a critical plane-based multiaxial fatigue criterion,
            Int J Fat, 2014

        """
        # psd is multiple-point
        if self.multipoint:
            self._all_points(criterion=criteria._cs, s_af=s_af, tau_af=tau_af, T=T)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq = criteria._cs(self,s=psd, s_af=s_af, tau_af=tau_af, T=T)
            self.set_eq_stress(eq_psd=s_eq, f=f)

    def multiaxial_rainflow(self, k=None):
        """Converts the stress tensor at one node to equivalent,
        scalar psd stress, for use in frequency domain multiaxial rainflow.

        ONLY WORKS FOR BIAXIAL STRESSES: PSD MATRIX (f,3,3) - single point or (N,f,3,3) - multiple points

        One fixed unit vector c is used for the whole spectrum (Pitoiset and Preumont 2000,
        Eq. 9 and Sec. 3), and the equivalent PSD is c^T Re(S(f)) c (Eqs. 11-14). The article
        picks c as the combination of maximum damage, computed with the single moment method
        (Sec. 4.1). For the single moment damage, which grows with c^T M c where
        M = sum_f f^(2/k) Re S(f) df, the maximum is the top eigenvector of M.

        :param k: float, optional
            Slope of the S-N curve, N*S^k = C. If given, c is chosen by maximum single moment
            damage (article method). If missing, c is the maximum-variance direction
            (top eigenvector of sum_f Re S(f) df, the limit k -> infinity) and a UserWarning
            asks for k.

        The chosen c is stored in `multiaxial_rainflow_direction`, shape (3,) or (N, 3).
        The sign of c does not matter (c and -c give the same PSD).

        --------

        - Pitoiset, Xavier, and André Preumont; 
            Spectral methods for multiaxial random fatigue analysis of metallic structures, 
            International journal of fatigue, 2000

        """
        # psd is multiple-point
        if self.multipoint:
            self._all_points(criterion=criteria._multiaxial_rainflow, k=k)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq = criteria._multiaxial_rainflow(self,s=psd, k=k)
            self.set_eq_stress(eq_psd=s_eq, f=f)

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
            self._all_points(criterion=criteria._thermoelastic)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq = criteria._thermoelastic(self,s=psd)
            self.set_eq_stress(eq_psd=s_eq, f=f)

    def liwi(self):
        """Converts the stress tensor at one node to equivalent
        stress, using the LiWI approach.

        ONLY WORKS WITH BIAXIAL AMPLITUDE SPECTRUM (f,3) - single point or (N,f,3) - multiple points

        The amplitude is the Bonte amplitude of Schmidt and Pandiya (2021), Eqs. 10-11. It equals
        `coin_liwi(k_a=sqrt(3), k_phi=0.5)` for plane stress. The LiWI of Schmidt and Kraft (2023), Eq. 38,
        is the revised definition with k_phi = 1, i.e. `coin_liwi(k_a=sqrt(3), k_phi=1)`.

        Amplitude convention: PSD = |A|**2/df, A is the RMS amplitude per frequency line. For one-sided peak
        amplitudes (FFT amplitudes, Schmidt and Pandiya 2021) pass A/sqrt(2).

        --------

        - Alexander T. Schmidt, Nimish Pandiya,
            Extension of the static equivalent stress hypotheses to linearly vibrating systems using wave interference – The LiWI approach,
            International Journal of Fatigue, 2021,

        """
        if not hasattr(self,'multiaxial_amplitude_spectrum'):
            raise Exception('Input Error. LiWI approach only works with multiaxial amplitude spectrum.')
        
        self._amplitude_criterion(criteria._liwi)


    def coin_liwi(self, k_a, k_phi):
        """Converts the stress tensor at one node to equivalent
        stress, using the COIN-LiWI method.

        ONLY WORKS WITH 3D AMPLITUDE SPECTRUM (f,6) - single point or (N,f,6) - multiple points
        
        Amplitude convention: PSD = |A|**2/df, A is the RMS amplitude per frequency line. For one-sided peak
        amplitudes (FFT amplitudes, Schmidt and Kraft 2023) pass A/sqrt(2).

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
        
        self._amplitude_criterion(criteria._coin_liwi, k_a=k_a, k_phi=k_phi)


    def EVMS_out_of_phase(self):
        """Converts the stress tensor at one node to equivalent
        stress, using the phase-sensitive EVMS criterion, adjusted for out-of-phase components.

        ONLY WORKS WITH BIAXIAL PSD (f,3,3) - single point or (N,f,3,3) - multiple points.
        A 3D PSD (f,6,6) raises ValueError.
               
        --------

        - Bonte, Martijn HA and de Boer, Andries and Liebregts R
            Determining the von Mises stress power spectral density for frequency domain fatigue analysis including out-of-phase stress components,
            Journal of Sound and Vibration, 2007
        
        """
        
        # psd is multiple-point
        if self.multipoint:
            self._all_points(criterion=criteria._EVMS_out_of_phase)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq_psd = criteria._EVMS_out_of_phase(self,s=psd)

            self.set_eq_stress(eq_psd=s_eq_psd, f=f)


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
            self._all_points(criterion=criteria._Nieslony, s_af=s_af, tau_af=tau_af, coefficient_load_type=coefficient_load_type)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq = criteria._Nieslony(self,s=psd, s_af=s_af, tau_af=tau_af, coefficient_load_type=coefficient_load_type)
            self.set_eq_stress(eq_psd=s_eq, f=f)

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
            self._all_points(criterion=criteria._Lemaitre, poisson_ratio=poisson_ratio)
        
        # psd is single-point
        else:
            psd, f = self._psd_matrix()
            s_eq = criteria._Lemaitre(self,s=psd, poisson_ratio=poisson_ratio)
            self.set_eq_stress(eq_psd=s_eq, f=f)