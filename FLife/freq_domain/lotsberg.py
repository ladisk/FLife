import numpy as np
from. narrowband import Narrowband

class Lotsberg(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Lotsberg[1].
      
    References
    ----------
    [1] Inge Lotsberg. Background for Revision of DNV-RP-C203 Fatigue Analysis of
        Offshore Steel Structure. 24th International Conference on Offshore Mechanics 
        and Arctic Engineering, Halkidiki, Greece, 2005. ASME, Paper No. OMAE2005–67549.
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
    >>> lb = FLife.Lotsberg(sd, PSD_splitting=('userDefinedBands', [80,150]))
    >>> print(f'Fatigue life: {lb.get_life(C,k):.3e} s.')

    Plot segmentated PSD, used in Lotsberg method

    >>> lower_band_index, upper_band_index= lb.band_stop_indexes
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

        :param spectral_data: Instance of class SpectralData
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
        """Calculate fatigue life with parameters C, k, as defined in [2].
        
        :param C: [int,float]
            S-N curve intercept [MPa**k].
        :param k: [int,float]
            S-N curve inverse slope [/].
        :return:
            Estimated fatigue life in seconds.
        :rtype: float
        """
        moments = self.spectral_data.get_spectral_moments(self.PSD_splitting, moments=[0])
        m0LF, = moments[0] #spectral moments for lower band
        m0HF, = moments[1] #spectral moments for upper band

        # -- positive slope zero crossing frequency
        v0LF, v0HF = self.spectral_data.get_nup(self.PSD_splitting)

        # -- damage intensity
        dNB_HF = self.damage_intesity_NB(m0=m0HF, nu=v0HF, C=C, k=k)
        dNB_LF = self.damage_intesity_NB(m0=m0LF, nu=v0LF, C=C, k=k) 

        d = dNB_HF*(1 - v0LF/v0HF) + v0LF*((dNB_HF/v0HF)**(1/k) + (dNB_LF/v0LF)**(1/k))**k
        T = 1/d
        
        return T

    def get_PDF(self, s):
        raise Exception(f'Function <get_PDF> is not available for class {self.__class__.__name__:s}.')