from ast import If
import numpy as np
from scipy import signal
from scipy import stats
from .tools import PSDgen, random_gaussian
import warnings

class SpectralData(object):
    """
    SpectralData object contains data, based on input signal time-history:
    power spectral density (PSD), spectral moments, spectral band estimators
    and others parameters, required for subsequent fatigue-life estimation [1].

    Notes
    -----
    Fatigue-life estimators inherit from instance of class SpectralData, so a 
    primitive caching is used. With class instantiation, values that are expected
    to be used subseqeuntly are calculated. The properties are created and then 
    called. Calculation is done only at the beginning and only once for following
    spectral method:

    - :class:`~FLife.freq_domain.Narrowband`
    - :class:`~FLife.freq_domain.WirschingLight`
    - :class:`~FLife.freq_domain.OrtizChen`
    - :class:`~FLife.freq_domain.Alpha075`
    - :class:`~FLife.freq_domain.TovoBenasciutti`
    - :class:`~FLife.freq_domain.Dirlik`
    - :class:`~FLife.freq_domain.ZhaoBaker`
    - :class:`~FLife.freq_domain.Park`
    - :class:`~FLife.freq_domain.JunPark`

    For multimodal fatigue-life estimators, spectral data are calculated a 
    second time, referring to each mode independently: 

    - :class:`~FLife.freq_domain.JiaoMoan`
    - :class:`~FLife.freq_domain.SakaiOkamura`
    - :class:`~FLife.freq_domain.FuCebon`
    - :class:`~FLife.freq_domain.ModifiedFuCebon`
    - :class:`~FLife.freq_domain.Low`
    - :class:`~FLife.freq_domain.Low2014`
    - :class:`~FLife.freq_domain.GaoMoan`
    - :class:`~FLife.freq_domain.Lotsberg`
    - :class:`~FLife.freq_domain.HuangMoan`
    - :class:`~FLife.freq_domain.SingleMoment`
    - :class:`~FLife.freq_domain.BandsMethod`

    SpectralData is also taken as base for Rainflow class, the fatigue-life 
    reference:

    - :class:`~FLife.time_domain.Rainflow`

    References
    ----------
    [1] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020

    Example
    -------
    Import modules, define time- and frequency-domain data
    
    >>> import FLife
    >>> import pyExSi as es
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> # time-domain data
    >>> N = 2 ** 16  # number of data points of time signal
    >>> fs = 2048  # sampling frequency [Hz]
    >>> t = np.arange(0, N) / fs  # time vector
    >>> # frequency-domain data
    >>> M = N // 2 + 1  # number of data points of frequency vector
    >>> freq = np.arange(0, M, 1) * fs / N  # frequency vector
    >>> PSD_lower = es.get_psd(freq, 20, 60, variance = 5)  # lower mode of random process
    >>> PSD_higher = es.get_psd(freq, 100, 120, variance = 2)  # higher mode of random process
    >>> PSD = PSD_lower + PSD_higher # bimodal one-sided flat-shaped PSD

    Get Gaussian stationary signal, instantiate SpectralData object and plot PSD

    >>> rg = np.random.default_rng(123) # random generator seed
    >>> x = es.random_gaussian(N, PSD, fs, rg) # Gaussian stationary signal
    >>> sd = FLife.SpectralData(input=x, dt=1/fs) # SpectralData instance
    >>> plt.plot(sd.psd[:,0], sd.psd[:,1]) 
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('PSD')
    >>> print(f'Irregularity factor: {sd.alpha2:.4f}') # irregularity factor

    Calculate spectral width parameter (epsilon = (1 - alpha2**2)**.5), with
    alpha2 obtained using function :meth:`.SpectralData.get_bandwidth_estimator`

    >>> # one element is in returned array, as entire PSD is taken as one band
    >>> alpha2, = sd.get_bandwidth_estimator(sd.PSD_splitting, i=2)
    >>> epsilon = np.sqrt(1 - alpha2**2)
    >>> print(f'Spectral width parameter: {epsilon:.4f}')

    Define S-N curve parameters and get fatigue-life estimatate

    >>> C = 1.8e+22  # S-N curve intercept [MPa**k]
    >>> k = 7.3 # S-N curve inverse slope [/]
    >>> dirlik = FLife.Dirlik(sd) # Dirlik's fatigue-life estimator
    >>> tb = FLife.TovoBenasciutti(sd) #Tovo-Benasciutti's fatigue-life estimator
    >>> jm = FLife.JiaoMoan(sd, PSD_splitting=('userDefinedBands', [80,150]))  # Jiao-Moan's fatigue-life estimator
    >>> print(f'Dirlik: {dirlik.get_life(C,k):.3e} s.')   
    >>> print(f'Tovo-Benasciutti, method 1: {tb.get_life(C,k,method='method 1'):.3e} s.')
    >>> print(f'Jiao-Moan: {jm.get_life(C,k):.3e} s.')   
  
    Plot segmented PSD, used in Jiao-Moan method

    >>> lower_band_index, upper_band_index= jm.band_stop_indexes
    >>> plt.plot(sd.psd[:,0], sd.psd[:,1])
    >>> plt.vlines(sd.psd[:,0][lower_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> # lower band
    >>> plt.fill_between(sd.psd[:lower_band_index,0], sd.psd[:lower_band_index,1], 'o', label='lower band', alpha=.2, color='blue')
    >>> plt.vlines(sd.psd[:,0][upper_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> # upper band
    >>> plt.fill_between(sd.psd[lower_band_index:upper_band_index,0], sd.psd[lower_band_index:upper_band_index,1], 'o', label='upper band', alpha=.5, color ='orange')
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('PSD')
    >>> plt.xlim(0,300)
    >>> plt.legend()

    Calculate irregularity factor for both PSD bands, defined in Jiao-Moan method

    >>> alpha2_lower_band, alpha2_higher_band = jm.spectral_data.get_bandwidth_estimator(jm.PSD_splitting, i=2) #or sd.get_bandwidth_estimator(jm.PSD_splitting, i=2)
    >>> print(f'Irregularity factor, lower band: {alpha2_lower_band:.4f}')
    >>> print(f'Irregularity factor, higher band:{alpha2_higher_band:.4f}')
    """

    def __init__(self, input=None, window='hann', nperseg=1280,
                 noverlap=None, psd_trim_length=None, 
                 T=None, fs=None, rg=None,
                 **kwargs):

        """Call parent constructor, analyse input and define fatigue life parameters and constants.
        
        :param input: str, tuple
                Defaults to 'GUI', where PSD is provided via GUI. Another option is dictionary,
                where input can be provided as psd with keys: 'PSD'(PSD array) and 'f'(frequency vector array),
                or as time history with keys: 'time_history' and 'dt', or as amplitude spectrum with keys: 'amplitude_spectrum' and 'f'.
                
                For uniaxial PSD input, PSD array should be the shape of (f), 
                for multiaxial PSD input, PSD array should be a matrix with the shape of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state.
                For multi-point PSD input (whole FEM model), PSD array should be a matrix
                with the shape of (N,f,6,6) for 3D stress state or (N,f,3,3) for 2D stress state, 
                where N is the number of points on the model.
                
                For uniaxial amplitude spectrum input, amplitude_spectrum array should be the shape of (f), 
                for multiaxial amplitude spectrum input, amplitude_spectrum array should be a matrix
                with the shape of (f,6) for 3D stress state or (f,3) for 2D stress state.
                For multi-point amplitude spectrum input (whole FEM model), amplitude_spectrum array should be a matrix
                with the shape of (N,f,6) for 3D stress state or (N,f,3) for 2D stress state, 

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
                or by PSD or by amplitude spectrum. If T and fs are provided, time histoy is generated.
                Defaults to None.
        :param fs: int, float, optional
                Sampling frequency of time history when random process is defined by `input` parameter 
                'GUI' or by PSD or by amplitude spectrum. If T and fs are provided, time histoy is generated.
                Defaults to None.
        :param rg: numpy.random._generator.Generator, optional
                Random generator controls phase of generated time history, when `input` is 'GUI' or 
                (PSD, frequency vector).
                Defaults to None.

        Attributes
        ----------
        data: array 1D
                Signal values with time spacing dt
        dt: float
                Time between discreete signal values    
        moments: array 1D
                Spectral moments from m0 to m4    
        psd: array 2D
                Normalized power spectral density
        t: float
                Length of signal in seconds, N * dt
        PSD_splitting: tuple
                PSD_splitting[0] is PSD spliting method, PSD_splitting[1] is method argument. 
                Splitting methods:

                - 'userDefinedBands', PSD_splitting[1] must be of type list or tupple, with N 
                  elements specifying upper band frequencies of N random processes.
                - 'equalAreaBands', PSD_splitting[1] must be of type int, specifying N random processes.

                Defaults to ('equalAreaBands', 1).

        Raises
        ------
        ValueError, for unknown input.
        """
        if 'dt' in kwargs:
            warnings.warn('Parameter `dt` has been deprecated since version 1.2 and will be removed in the future.' + 
            '\nSampling interval of time signal should be given as second element of tuple `input`.')
            input = (input, kwargs['dt'])
        
        # Default input is by GUI
        if input == None or input == 'GUI':
            self.psd = PSDgen().get_PSD()
            
            # needed parameters for time-history generation
            if T is not None and fs is not None:
                self._set_time_history(f=self.psd[:,0], psd=self.psd[:,1], T=T, fs=fs, **kwargs)


        # If input is tuple, dictionary is created (backwards compatibility)
        elif isinstance(input, tuple) and len(input) == 2:
            warnings.warn('Tuple input has been deprecated. Use dictionary input instead.')
            # If input is time signal
            if isinstance(input[0], np.ndarray) and isinstance(input[1], (int, float)):
                input = {'time_history': input[0], 'dt': input[1]}
            # or PSD
            elif isinstance(input[0], np.ndarray) and isinstance(input[1], np.ndarray):
                input = {'PSD': input[0], 'f': input[1]}

            elif isinstance(input[0], str) and isinstance(input[1], (int, float)):
                    warnings.warn('Path to file option has been deprecated since version 1.2 and will be removed in the future.')
                    self.data = self._readf(input[0])
                    self.dt = input[1] # Sampling interval
                    self.t = self.dt * self.data.size # length of signal time-history
                    self.psd = self._calculate_psd(self.data, fs=1.0/self.dt, window=window,
                                                nperseg=nperseg, noverlap=noverlap,
                                                trim=psd_trim_length)
                                                
                    

        # Input is dictionary a) PSD and frequency vector or b) time history and sampling interval
        if isinstance(input, dict):
            if 'time_history' in input and 'dt' in input:
                self.data = input['time_history']
                self.dt = input['dt']
                self.t = self.dt * self.data.size
                self.psd = self._calculate_psd(self.data, fs=1.0/self.dt, window=window,
                                            nperseg=nperseg, noverlap=noverlap,
                                            trim=psd_trim_length)
            elif 'PSD' in input and 'f' in input:
                #multiaxial PSD
                if input['PSD'].ndim>1:
                    self.multiaxial_psd = (input['PSD'],input['f'])

                    # Check dimension of multiaxial stress PSD
                    if self.multiaxial_psd[0].ndim == 3 or self.multiaxial_psd[0].ndim == 4 and (
                        (self.multiaxial_psd[0].shape[-1] == 6 and self.multiaxial_psd[0].shape[-2] == 6) or
                        (self.multiaxial_psd[0].shape[-1] == 3 and self.multiaxial_psd[0].shape[-2] == 3)
                    ):     
                        if self.multiaxial_psd[0].ndim == 4:
                            self.multipoint = True

                        if T is not None and fs is not None:
                            self.t = T
                            self.fs = fs
                    else:
                        raise Exception('Input Error. PSD matrix should be the size of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state')

                    if T is not None and fs is not None:
                        self._check_fs(f=input['f'], psd=input['PSD'], T=T, fs=fs, multipoint=True, **kwargs)
            
                # Uniaxial PSD
                elif input['PSD'].ndim==1:
                    psd = input['PSD']
                    f = input['f']
                    self.psd = np.column_stack((f, psd))
                    # needed parameters for time-history generation
                    if T is not None and fs is not None:
                        self._set_time_history(f=f, psd=psd, T=T, fs=fs, **kwargs)
            
            elif 'amplitude_spectrum' in input and 'f' in input:
                #multiaxial amplitude spectrum
                if input['amplitude_spectrum'].ndim>1:
                    self.multiaxial_amplitude_spectrum = (input['amplitude_spectrum'],input['f'])

                    # Check dimension of multiaxial amplitude spectrum
                    if self.multiaxial_amplitude_spectrum[0].ndim == 2 or self.multiaxial_amplitude_spectrum[0].ndim == 3 and (
                        (self.multiaxial_amplitude_spectrum[0].shape[-1] == 6) or
                        (self.multiaxial_amplitude_spectrum[0].shape[-1] == 3)
                    ):     
                        if self.multiaxial_amplitude_spectrum[0].ndim == 3:
                            self.multipoint = True

                        if T is not None and fs is not None:
                            self.t = T
                            self.fs = fs
                    else:
                        raise Exception('Input Error. PSD matrix should be the size of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state')

                    if T is not None and fs is not None:
                        self._check_fs(f=input['f'], psd=input['amplitude_spectrum'], T=T, fs=fs, multipoint=True, **kwargs)
                
                # Uniaxial amplitude spectrum
                elif input['amplitude_spectrum'].ndim==1:
                    print('Input amplitude spectrum is uniaxial')
                    amplitde_spectrum = input['amplitude_spectrum']
                    f = input['f']
                    psd = amplitde_spectrum**2/(f[1]-f[0])
                    self.psd = np.column_stack((f, psd))
                    # needed parameters for time-history generation
                    if T is not None and fs is not None:
                        self._set_time_history(f=f, psd=psd, T=T, fs=fs, **kwargs)

            else:
                raise Exception('Unrecognized Input Error. `input` should be dictionary with keys `PSD` and `f` or `time_history` and `dt`.')

        else:
            raise Exception('Unrecognized Input Error') 

        if hasattr(self,'psd'):
            self.PSD_splitting = ('equalAreaBands', 1) 
            self._calculate_coefficients()

    def _readf(self, filename):
        """Read input file and extract values in form of array (float).
        
        Values inside txt file must be separated with whitespace, e.g. space or newline.
        
        Parameters
        ----------
        filename : string
                   path to file
        
        Returns
        -------
        1D array filled with values. 
        """
        f = open(filename)
        data = np.array(map(float,f.read().split()))
        f.close()
    
        return data
    
    def _check_fs(self, f, psd, T=None, fs=None, multipoint=False, **kwargs):
        
        #if input is multi point, multiaxial
        if multipoint:
            f_max_indx = np.where(psd>0)[1][-1]
        else:
            f_max_indx = np.where(psd>0)[0][-1]
        f_max = f[f_max_indx]
                
        # Sampling frequency check
        if fs < 2*f_max:
            raise Exception('Parameter `fs` should be higher. It should be fs >= 2*fmax.')
                    
        if fs < 10*f_max:
            warnings.warn(f'Parameter `fs` should be higher. It is suggested to use fs>=10*fmax.')

    def _set_time_history(self, f, psd, T=None, fs=None, **kwargs):
        """Generates and set time-history of signal on basis of PSD and frequency vector.
        """   
        if T is not None and fs is not None:
            
            self._check_fs(f, psd, T, fs, multipoint=False, **kwargs)
            # get time history
            time, signal = random_gaussian(freq=f, PSD=psd, T=T, fs=fs, **kwargs) # stationary, normally distributed

            #set data
            self.data = signal
            self.dt = 1.0/fs  # Sampling interval
            self.t = T # signal time-history duration
            
        else:
            warnings.warn('Parameters `T` and `fs` should both be specified. Time history is not set.')

    def _calculate_psd(self, data, fs=1, window='hamming', nperseg=10280, noverlap=None, trim=None):
        """Calculates PSD using welch estimator.
        """
        f, p = signal.welch(data, fs=fs, window=window,
                        nperseg=nperseg, noverlap=noverlap)

        psd = np.vstack((f, p)).transpose()

        if trim is None:
            return psd
        else:
            df = fs / nperseg
            trim_idx = int(np.floor(trim * df))

            return psd[:trim_idx, :trim_idx]

    def _calculate_coefficients(self):
        """Calculates coefficients, needed in spectral-fatigue-life estimates. Coefficients are 
        calculated with regard to entire PSD.
        """
        self.moments = np.ravel(self.get_spectral_moments(self.PSD_splitting, moments=[0,1,2,3,4]))
        self.nu, = self.get_nup(self.PSD_splitting)
        self.m_p, = self.get_mp(self.PSD_splitting)
        self.alpha075, = self.get_bandwidth_estimator(self.PSD_splitting, i=0.75)
        self.alpha1, = self.get_bandwidth_estimator(self.PSD_splitting, i=1)
        self.alpha2, = self.get_bandwidth_estimator(self.PSD_splitting, i=2)

    def _get_spectral_moment(self, psd, i):
        """Calculates i-th spectral moment. 
        """
        if not isinstance(i, (int, float)):
            raise TypeError('Parameter `i` must be of type int or float')

        f = psd[:, 0]
        p = psd[:, 1]
        
        if np.__version__>='2.0.0':
            trapezoid = np.trapezoid
        else:
            trapezoid = np.trapz
        
        return trapezoid((2*np.pi*f)**i * p, f)


    def get_spectral_moments(self, PSD_splitting=None, moments=[0,1,2,3,4]):
        """Returns spectral moments, specified by `moments`.
        Depending on parameter `PSD_splitting`, function returns
        an array of shape (M x N), where M is the number of PSD segments
        and N is length of parameter `moments`. For SpectralData, M defaults
        to 1.

        :param PSD_splitting: :attr:`.PSD_splitting`
        :param moments: list
            List of spectral momemnts to be calculated. Its elements must be of type
            int or float.
        :return: Spectral moments.
        :rtype: numpy.ndarray; 
            An array object of shape (M x N).
        """
        if PSD_splitting == None:
            PSD_splitting = self.PSD_splitting 
        band_stop_indexes = self._get_band_stop_frequency(PSD_splitting)
        m_list = list()

        for index in range(len(band_stop_indexes)):
            if index == 0:
                m_list.append([self._get_spectral_moment(self.psd[:band_stop_indexes[index]+1,:], i) for i in moments])
            else:
                m_list.append([self._get_spectral_moment(self.psd[band_stop_indexes[index-1]:band_stop_indexes[index]+1,:], i) for i in moments])

        return np.array(m_list)

    def get_bandwidth_estimator(self, PSD_splitting=None, i=2): 
        """Calculates bandwidth estimator alpha_i [1]. Takes parameter `PSD_splitting`
        for reference to PSD segmentation.

        :param PSD_splitting: :attr:`.PSD_splitting`
        :param i: [int,float]
            Bandwith estimator index; alpha_i = m_i/((m0*m_2i)**.5), where m_i represents
            i-th spectral moment.
        :return: bandwidth estimator;
            An array object of length N, containing bandwidth estimator for N bands.
        :rtype: numpy.ndarray
        """
        if PSD_splitting == None:
            PSD_splitting = self.PSD_splitting 

        if not isinstance(i, (int, float)):
            raise TypeError('Parameter `i` must be of type int or float')
        
        moments = self.get_spectral_moments(PSD_splitting, moments=[0,i,2*i])
        alpha_list = list()
        for band_moments in moments:
            alpha_i  = band_moments[1] / np.sqrt(band_moments[0] * band_moments[2])
            alpha_list.append(alpha_i)

        return np.array(alpha_list)

    def get_vanmarcke_parameter(self, PSD_splitting=None):
        """Calculates Vanmarcke bandwidth parameter epsilon_V[1]. Takes parameter 
        `PSD_splitting` for reference to PSD segmentation.

        :param PSD_splitting: :attr:`.PSD_splitting`
        :return: bandwidth estimator;
            An array object of length N, containing vanmarcke's parameter for N bands.
        :rtype: numpy.ndarray
        """
        if PSD_splitting == None:
            PSD_splitting = self.PSD_splitting 

        alpha = self.get_bandwidth_estimator(PSD_splitting, i=1)
        epsV_list = list()
        for alpha_1 in alpha:
            epsV_i  = np.sqrt(1 - alpha_1**2)
            epsV_list.append(epsV_i)

        return np.array(epsV_list)

    def get_nup(self, PSD_splitting=None):
        """Calculates nu_p; expected frequency of positive slope zero crossing [1].
        Takes parameter `PSD_splitting` for reference to PSD segmentation.

        :param PSD_splitting: :attr:`.PSD_splitting`
        :return: bandwidth estimator;
            An array object of length N, containing nup for N bands.
        :rtype: numpy.ndarray
        """
        moments = self.get_spectral_moments(PSD_splitting, moments=[0,2])
        nup_list = list()
        for band_moments in moments:
            nup_i  = 1/(2*np.pi) * np.sqrt(band_moments[1] / band_moments[0])
            nup_list.append(nup_i)

        return np.array(nup_list)

    def get_mp(self, PSD_splitting=None):
        """Calculates m_p; expected peak frequency [1]. Takes parameter 
        `PSD_splitting` for reference to PSD segmentation.

        :param PSD_splitting: :attr:`.PSD_splitting`
        :return: bandwidth estimator;
            An array object of length N, containing m_p for N bands.
        :rtype: numpy.ndarray
        """
        moments = self.get_spectral_moments(PSD_splitting, moments=[2,4])
        mp_list = list()
        for band_moments in moments:
            mp_i  = 1/(2*np.pi) * np.sqrt(band_moments[1] / band_moments[0])
            mp_list.append(mp_i)

        return np.array(mp_list)

    def get_peak_PDF(self, s):
        """Returns peak amplitude PDF(Probability Density Function) as a function of stress s [1].

        :param s:  numpy.ndarray
            Stress vector.
        :return: function pdf(s)
        """
        m0 = self.moments[0]
        alpha2 = self.alpha2

        def pdf(s):
            px = np.sqrt(1.0 - alpha2**2)/np.sqrt(2.0 * np.pi * m0) * \
                np.exp( - (s**2) / (2.0 * m0 * (1.0 - alpha2**2))) +\
                alpha2*s/m0 * np.exp( - (s**2) / (2*m0)) * \
                stats.norm.cdf((alpha2 * s) / (np.sqrt(m0 * (1 - alpha2**2))))
            return px
        return pdf(s)

    def _get_band_stop_frequency(self, PSD_splitting=None):
        """Returns stop band frequency indexes of segmentated PSD. Takes parameter 
        `PSD_splitting` for reference to PSD segmentation.
        
        :param PSD_splitting: tuple;
            splitting[0] is PSD spliting method, splitting[1] is method argument. 
            Splitting methods:
            - `userDefinedBands`, PSD_splitting[1] must be of type list or tupple,
              with N elements specifying upper band frequencies of N random processes.
            - `equalAreaBands`, PSD_splitting[1] must be of type int, specifying N
              random processes.
        :return freq_indx: tuple
            Upper band frequency indexes of segmentated PSD.
        """
        if PSD_splitting == None:
            PSD_splitting = self.PSD_splitting 

        method_dict = {'equalAreaBands': self._equalAreaBands,
                        'userDefinedBands': self._userDefinedBands
                    }

        if isinstance(PSD_splitting, tuple):
            method = PSD_splitting[0]
            arg = PSD_splitting[1]
            try:
                freq_indx = method_dict[method](arg)
            except:
                raise ValueError(f'Unknown splitting method: {method:s}')
        else:
            raise TypeError('Parameter `PSD_splitting` must be of type tuple.') 
    
        return freq_indx

    def _userDefinedBands(self, freq_list):
        #data type checking
        if isinstance(freq_list, (list, tuple)):
            pass
        else:
            raise TypeError('Parameter `PSD_splitting[1]` must be of type list or tuple.')

        for freq in freq_list:
            if isinstance(freq, (int,float)):
                pass
            else:
                raise TypeError('Parameter `PSD_splitting[1]` elements must be of type int or float.')

        psd_freq = self.psd[:,0]
        freq_indx = list()
        for n in range(len(freq_list)):
            freq_indx.append(np.abs(psd_freq - freq_list[n]).argmin())

        return tuple(freq_indx)

    def _equalAreaBands(self, N):
        if isinstance(N, int):
            pass
        else:
            raise TypeError('Parameter `PSD_splitting[1]` must be of type int.')

        psd_val = self.psd[:,1]
        q_area = np.sum(psd_val) # (not the REAL area because x-axis unit is omitted)
        
        # -- Calculate a cumulative sum vector of psd
        c_psd = np.cumsum(psd_val)
        
        # -- Find where cumulative sum is equal to 1/N, 2/N, ...,  N/N of Area
        freq_indx = list()
        for n in range(N):
            freq_indx.append(np.abs(c_psd - (n+1) * q_area/N).argmin())

        return tuple(freq_indx)