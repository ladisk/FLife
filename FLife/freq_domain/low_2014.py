import numpy as np
from .narrowband import Narrowband
import warnings

class Low2014(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Low[1].
    
    References
    ----------
    [1] Ying Min Low. A simple surrogate model for the rainflow fatigue damage arising 
        from processes with bimodal spectra. Marine Structures, 38:72-88, 2014
    [2] Aleš Zorman and Janko Slavič and Miha Boltežar. 
        Vibration fatigue by spectral methods—A review with open-source support, 
        Mechanical Systems and Signal Processing, 2023, 
        https://doi.org/10.1016/j.ymssp.2023.110149
    
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

    Define S-N curve parameters and get fatigue-life estimatate

    >>> C = 1.8e+22  # S-N curve intercept [MPa**k]
    >>> k = 7.3 # S-N curve inverse slope [/]
    >>> low2014 = FLife.LowBimodal2014(sd, PSD_splitting=('userDefinedBands', [80,150]))
    >>> print(f'Fatigue life: {low2014.get_life(C,k):.3e} s.')

    Plot segmentated PSD, used in LowBimodal2014 method

    >>> lower_band_index, upper_band_index= low2014.band_stop_indexes
    >>> plt.plot(sd.psd[:,0], sd.psd[:,1])
    >>> plt.vlines(sd.psd[:,0][lower_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> plt.fill_between(sd.psd[:lower_band_index,0], sd.psd[:lower_band_index,1], 'o', label='lower band', alpha=.2, color='blue')
    >>> plt.vlines(sd.psd[:,0][upper_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> plt.fill_between(sd.psd[lower_band_index:upper_band_index,0], sd.psd[lower_band_index:upper_band_index,1], 'o', label='upper band', alpha=.5, color ='orange')
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('PSD')
    >>> plt.xlim(0,300)
    >>> plt.legend()
    """
    def __init__(self, spectral_data, PSD_splitting = ('equalAreaBands', 2)):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData       
        :param PSD_splitting: tuple
                PSD_splitting[0] is PSD spliting method, PSD_splitting[1] is method argument. 
                Splitting methods:

                - 'userDefinedBands', PSD_splitting[1] must be of type list or tupple, with N 
                  elements specifying upper band frequencies of N random processes.
                - 'equalAreaBands', PSD_splitting[1] must be of type int, specifying N random processes.

                Defaults to ('equalAreaBands', 2).
        """
        Narrowband.__init__(self, spectral_data)
        self.PSD_splitting = PSD_splitting
        self.band_stop_indexes = self.spectral_data._get_band_stop_frequency(self.PSD_splitting)
        
    def get_life(self, C, k):
        """Calculate fatigue life with parameters C, k, as defined in [1, 2].

        :param C: [int,float]
            S-N curve intercept [MPa**k].
        :param k: [int,float]
            S-N curve inverse slope [/].
        :param approximation: Boolean
            IF true, approximated PDF of large peaks is used for bimodal random process. 
        :return:
            Estimated fatigue life in seconds.
        :rtype: float
        """
        # central frequencies
        v0L, v0H = self.spectral_data.get_nup(self.PSD_splitting)
        beta = v0H/v0L

        # spectral moments
        moments = self.spectral_data.get_spectral_moments(self.PSD_splitting, moments=[0,2])
        m0L, m2L = moments[0] #spectral moments for lower band
        m0H, m2H = moments[1] #spectral moments for upper band

        # -- normalized variances
        m0 = np.sum(moments[:, 0])
        m0L_norm = m0L/m0
        m0H_norm = m0H/m0
        
        #check method validity range
        if not 3 < beta < np.inf:
            warnings.warn(f'Correction factor is optimized for zero upcrossing rates ratio 3 <= `beta` < infinity. Actual value is `beta`= {beta:.2f}. Results should be evaluated carefully.')
        if not 3 <= k <= 8:
            warnings.warn(f'Correction factor is optimized for 3 <= `k` <= 8. Results should be evaluated carefully.')

        # Correction factor R
        b1 = (1.111 + 0.7421*k - 0.0724*k**2) * beta**(-1) + (2.403 - 2.483*k) * beta**(-2)
        b2 = (-10.45 + 2.65*k ) * beta**(-1) + (2.607 + 2.63*k - 0.0133*k**2) * beta**(-2)
        L = (b1*np.sqrt(m0H_norm) + b2*m0H_norm - (b1+b2)*m0H_norm**(3/2) + m0H_norm**(k/2)) * (beta-1) + 1
        R = L/(np.sqrt(1 - m0H_norm + beta**2 * m0H_norm))

        # narrowband damage intensity
        v0 = 1/(2*np.pi) * np.sqrt((m2L + m2H)/(m0L + m0H))
        d_NB = self.damage_intesity_NB(m0=m0, nu=v0, C=C, k=k) 

        # damage intensity
        d = d_NB * R
        T = 1/d

        return T
            
    def get_PDF(self, s):
        raise Exception(f'Function <get_PDF> is not available for class {self.__class__.__name__:s}.')