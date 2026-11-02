import warnings
import numpy as np
from . import criteria
from ..spectralData import SpectralData

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
    - gao_mwcm: Spectral Modified Wöhler Curve Method (Gao et al. 2021)
    - EVMS: Equivalent von Misses stress
    - cs: Carpinteri-Spagnoli criterion
    - multiaxial_rainflow: Frequency-based multiaxial rainflow criterion
    - thermoelastic: Thermoelasticity-based criterion
    - EVMS_out_of_phase: EVMS adaptation for out-of-phase components
    - Nieslony: Nieslony criterion, combining von Mises and hydrostatic stresses
    - Lemaitre: Equivalent Lemaitre stress

    For a review of the implemented multiaxial criteria, see:
    Jaša Šonc and Janko Slavič, Review of Multiaxial Vibration Fatigue Spectral
    Methods—With Open-Source Support, Applied Nonlinear Dynamics and Vibrations,
    2026. https://doi.org/10.53941/andv.2026.100003

    """

    def __init__(self, input=None, window='hann', nperseg=1280,
                 noverlap=None, psd_trim_length=None, 
                 T=None, fs=None, rg=None,
                 **kwargs):
        """
        Class constructor for EquivalentStress object.

        :param input: dictionary or instance of SpectralData class.
                Input data for EquivalentStress object. If dictionary, it should contain keys: `PSD` and `f`.
                PSD should be an array with shape (f,6,6) for 3D stress state or (f,3,3) for 2D stress state.
                If multi-point PSD is provided, the shape should be (N,f,6,6) or (N,f,3,3) respectively.
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
                Length of the time history of the equivalent stress. If T and fs are provided,
                the time history is generated. Defaults to None.
        :param fs: int, float, optional
                Sampling frequency of the time history of the equivalent stress. If T and fs are
                provided, the time history is generated. Defaults to None.
        :param rg: numpy.random._generator.Generator, optional
                Random generator for the phase of the time history of the equivalent stress.
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
            if hasattr(input,'t') and hasattr(input,'fs'):
                self.t = input.t
                self.fs = input.fs
                self.rg = getattr(input, 'rg', None)
            self.multipoint = self.spectral_data.multipoint
        else:
            raise Exception('Unrecognized Input Error. `input` should be a dictionary with keys: `PSD` and `f`).')

    def loop_over_points(self, criterion, *args, **kwargs):
        """Deprecated, will be removed in a future version. The criterion methods
        (for example `max_normal` or `EVMS`) handle multiple points themselves; call them instead.

        Loop the selected criterion over multiple points on the model and set `eq_psd_multipoint`.
        """
        warnings.warn('`EquivalentStress.loop_over_points` is deprecated and will be removed in a future '
                      'version. Call the criterion method (for example `max_normal` or `EVMS`) instead; '
                      'it handles multiple points itself.', DeprecationWarning, stacklevel=2)
        self._each_point(criterion, *args, **kwargs)

    def _each_point(self, criterion, *args, **kwargs):
        """Apply a criterion for one point (f, ...) to each point in turn and set `eq_psd_multipoint`.
        Used by the criteria whose critical-plane search works on one point at a time."""
        s_eq_multi_point = np.empty((self.multiaxial_psd[0].shape[:2]))
        for i in range(len(s_eq_multi_point)):
            s_eq_multi_point[i] = criterion(self,s=self.multiaxial_psd[0][i],*args, **kwargs)
        self.eq_psd_multipoint = (s_eq_multi_point, self.multiaxial_psd[1]) #(multipoint psd, freq)

    def _apply(self, criterion, each_point=False, **kwargs):
        """Apply a criterion to the multiaxial PSD and set the equivalent stress.

        Single point: set the equivalent PSD with `set_eq_stress`. Multiple points: set
        `eq_psd_multipoint`, with `_each_point` if `each_point` is True (the criterion takes one
        point at a time), else with `_all_points`.
        """
        if not self.multipoint:
            s, f = self.multiaxial_psd
            self.set_eq_stress(eq_psd=criterion(self, s=s, **kwargs), f=f)
        elif each_point:
            self._each_point(criterion, **kwargs)
        else:
            self._all_points(criterion, **kwargs)

    def _all_points(self, criterion, *args, **kwargs):
        """Apply a criterion that accepts all points at once (N, f, ...) and set `eq_psd_multipoint`."""
        psd, f = self.multiaxial_psd
        self.eq_psd_multipoint = (criterion(self, s=psd, *args, **kwargs), f) #(multipoint psd, freq)

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
                Search method for optimization. Options are 'local' or 'global'. Local is prefered, unless the optimization fails.
        
        --------

        - Nieslony, Adam and Macha, Ewald (2007);
            Spectral method in multiaxial random fatigue

        """
        self._apply(criteria._max_normal, each_point=True, search_method=search_method)


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
        self._apply(criteria._max_shear, each_point=True, search_method=search_method)

    def max_normal_and_shear(self, s_af, tau_af, search_method='local'):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using method of maximum normal and shear stress.
        
        Critical plane is based on max variance of shear stress
        
        :param s_af: float
            Fully reversed bending-fatigue limit. Used for calculating material fatigue coefficient K [1].
        :param tau_af: float
            Fully reversed torsion-fatigue limit. Used Used for calculating material fatigue coefficient K [1].
        :param search_method: str, optional, default 'local'
                Search method for optimization. Options are 'local' or 'global', see `max_normal`.

        --------
            
        - Nieslony, Adam and Macha, Ewald (2007);
            Spectral method in multiaxial random fatigue

        - Matjaz Mrsnik, Janko Slavic and Miha Boltezar
            Multiaxial Vibration Fatigue A Theoretical and Experimental Comparison.
            Mechanical Systems and Signal Processing, 2016

        """
        self._apply(criteria._max_normal_and_shear, each_point=True, search_method=search_method, s_af=s_af, tau_af=tau_af)

    def gao_mwcm(self, s_af, tau_af, search_method='local'):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using the spectral Modified Wöhler Curve Method (MWCM) of Gao et al. [1].

        Critical plane is the plane of maximum shear stress variance [1, Eq. 12]. On it, the
        equivalent stress is tau_eq = tau_c + q sigma_c [1, Eqs. 16-18], with tau_c the shear
        stress and sigma_c the normal stress, and its PSD is
        G_eq = G_tau + 2 q Re(G_tau,sigma) + q^2 G_sigma [1, Eqs. 25-27].

        q = 1 - s_af / (2 tau_af) follows Susmel and Lazzarin [2, Eq. 11]. Eq. 14 in [1] has
        sigma_-1 and tau_-1 swapped, which gives q = s_af / tau_af - 1/2; with that value a fully
        reversed uniaxial load does not reach its fatigue limit s_af.
        The linear form [1, Eq. 16] is exact for pure torsion. For uniaxial load at s_af it differs
        from [2, Eq. 11] by 1.8 % for s_af / tau_af = sqrt(3).

        Only zero mean stress is supported (p = 1 in [1, Eqs. 19-20]). The Goodman mean stress
        correction of [1] needs a mean stress input, which is not available yet.

        tau_eq is a shear stress. As for the other criteria, the result is returned as an
        equivalent normal stress sigma_eq = (s_af / tau_af) tau_eq, so that it is used with the
        fully reversed axial S-N curve, as in [1, Eq. 30]. Pure torsion tau_af and uniaxial
        s_af both give sigma_eq = s_af (the latter within the linearisation above).

        :param s_af: float
            Fully reversed axial fatigue limit sigma_-1.
        :param tau_af: float
            Fully reversed torsion fatigue limit tau_-1.
        :param search_method: str, optional, default 'local'
                Search method for optimization. Options are 'local' or 'global', see `max_normal`.

        --------

        - [1] Gao D, Yao W, Wen W, Huang J;
            A multiaxial fatigue life prediction method for metallic material under combined
            random vibration loading and mean stress loading in the frequency domain,
            International Journal of Fatigue, 2021, https://doi.org/10.1016/j.ijfatigue.2021.106235

        - [2] Susmel L, Lazzarin P;
            A bi-parametric Wöhler curve for high cycle multiaxial fatigue assessment,
            Fatigue & Fracture of Engineering Materials & Structures, 2002, https://doi.org/10.1046/j.1460-2695.2002.00462.x

        """
        self._apply(criteria._gao_mwcm, each_point=True, search_method=search_method, s_af=s_af, tau_af=tau_af)

    def EVMS(self):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using ther equivalent von Misses stress in frequency domain (EVMS)
    
        --------

        - Preumont, A., & Piéfort, V. (1994);
            Predicting Random High-Cycle Fatigue Life With Finite Elements.

        """
        self._apply(criteria._EVMS)

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
            T given to EquivalentStress or SpectralData, else 1 s with a warning.
        
        --------

        - Carpinteri A, Spagnoli A and Vantadori S; 
            Reformulation in the frequency domain of a critical plane-based multiaxial fatigue criterion,
            Int J Fat, 2014

        """
        self._apply(criteria._cs, each_point=True, s_af=s_af, tau_af=tau_af, T=T)

    def multiaxial_rainflow(self, k=None):
        """Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, for use in frequency domain multiaxial rainflow.

        ONLY WORKS FOR BIAXIAL STRESSES: PSD MATRIX (f,3,3) - single point or (N,f,3,3) - multiple points

        One constant unit vector c is used for the whole spectrum (Pitoiset and Preumont 2000,
        Eqs. 9 and 10), and the equivalent PSD is c^T Re(S(f)) c (Eqs. 11-14). The article
        picks c as the combination of maximum damage (Sec. 3), computed with the single moment
        method (Sec. 4.1). The single moment damage grows with c^T M c, where
        M = int f^(2/k) Re S(f) df, so the maximum is the top eigenvector of M.

        :param k: float, optional
            Slope of the S-N curve, N*S^k = C. If given, c is chosen by maximum single moment
            damage (article method). If missing, c is the maximum-variance direction
            (top eigenvector of int Re S(f) df, the limit k -> infinity) and a UserWarning
            asks for k.

        The chosen c is stored in `multiaxial_rainflow_direction`, shape (3,) or (N, 3).
        The sign of c does not matter (c and -c give the same PSD).

        --------

        - Pitoiset, Xavier, and André Preumont; 
            Spectral methods for multiaxial random fatigue analysis of metallic structures, 
            International journal of fatigue, 2000

        - Lutes, Loren D., and Curtis E. Larsen;
            Improved spectral method for variable amplitude fatigue prediction,
            Journal of Structural Engineering, 1990

        """
        self._apply(criteria._multiaxial_rainflow, k=k)

    def thermoelastic(self):
        """Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using the thermoelasticity based criterion, 
        simulating the equivalent stress, detected with the thermal camera

        --------

        - Šonc J, Zaletelj K and Slavič J;
            Application of thermoelasticity in the frequency-domain multiaxial vibration-fatigue criterion, 
            Mechanical Systems and Signal Processsing, 2025

        """
        self._apply(criteria._thermoelastic)

    def EVMS_out_of_phase(self):
        """Converts the stress tensor at one node or multiple nodes to equivalent
        stress, using the phase-sensitive EVMS criterion, adjusted for out-of-phase components.

        ONLY WORKS WITH BIAXIAL PSD (f,3,3) - single point or (N,f,3,3) - multiple points.
        A 3D PSD (f,6,6) raises ValueError.
               
        --------

        - Bonte, Martijn HA and de Boer, Andries and Liebregts R
            Determining the von Mises stress power spectral density for frequency domain fatigue analysis including out-of-phase stress components,
            Journal of Sound and Vibration, 2007
        
        """
        self._apply(criteria._EVMS_out_of_phase)

    def Nieslony(self, s_af, tau_af, coefficient_load_type='tension'):
        """Converts the stress tensor at one node or multiple nodes to equivalent,
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
        self._apply(criteria._Nieslony, s_af=s_af, tau_af=tau_af, coefficient_load_type=coefficient_load_type)

    def Lemaitre(self, poisson_ratio):
        """
        Converts the stress tensor at one node or multiple nodes to equivalent,
        scalar psd stress, using ther equivalent Lemaitre stress in frequency domain
    
        --------

        - Jingran Ge, Yi Sun, Song Zhou;
            Fatigue life estimation under multiaxial random loading by means of the equivalent Lemaitre stress and multiaxial S–N curve methods.
            International Journal of Fatigue, 2015
            
        """
        self._apply(criteria._Lemaitre, poisson_ratio=poisson_ratio)
