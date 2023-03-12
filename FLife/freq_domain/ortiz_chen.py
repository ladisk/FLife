import numpy as np
from scipy.special import gamma
from .narrowband import Narrowband

class OrtizChen(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Ortiz and Chen [1].
   
    References
    ----------
    [1] K. Ortiz and N.K. Chen. Fatigue damage prediction for stationary wideband processes.
        Fifth international conference on applications of statistics and probability in soil 
        and structural engineering, 1987
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
    >>> oc = FLife.OrtizChen(sd)
    >>> print(f'Fatigue life: {oc.get_life(C,k):.3e} s.')
    """
    def __init__(self, spectral_data):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData
        """
        Narrowband.__init__(self, spectral_data)
                           
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
        m0 = self.spectral_data.moments[0]
        m2 = self.spectral_data.moments[2]

        m_2k, = self.spectral_data.get_spectral_moments(self.spectral_data.PSD_splitting, moments=[2/k])[0]
        m_2k_2, = self.spectral_data.get_spectral_moments(self.spectral_data.PSD_splitting, moments=[2/k + 2])[0]

        nu = self.spectral_data.nu
        alpha2 = self.spectral_data.alpha2

        dNB = self.damage_intesity_NB(m0=m0, nu=nu, C=C, k=k)
        ro = 1/alpha2 * np.sqrt((m2*m_2k)/(m0*m_2k_2))**k
        T = 1 / (ro * dNB)
        
        return T

    def get_PDF(self, s):
        raise Exception(f'Function <get_PDF> is not available for class {self.__class__.__name__:s}.')