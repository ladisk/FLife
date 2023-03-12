import numpy as np
from scipy.special import gamma
from .narrowband import Narrowband

class SingleMoment(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Lutes and Larsen[1, 2].
      
    References
    ----------
    [1] L.D. Lutes, C.E. Larsen. Improved spectral method for variable amplitude fatigue prediction,
        Journal of Structural Engineering ASCE, 116(4):1149-1164, 1990
    [2] C.E. Larsen, L.D. Lutes. Predicting the Fatigue Life of Offshore Structures by the Single-Moment Spectral Method,
        Probabilistic Engineering Mechanics, 6(2):96-108, 1991 
    [3] Aleš Zorman and Janko Slavič and Miha Boltežar. 
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
    >>> sm = FLife.SingleMoment(sd)
    >>> print(f'Fatigue life: {sm.get_life(C,k):.3e} s.')
    """
    def __init__(self, spectral_data):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData
        """     
        self.spectral_data = spectral_data

    def damage_intesity_SM(self, m_2k, C, k):
        """Calculates damage intensity with parameters m_2k, nu, C, k, as defined in [1,2].

        :param m_2k: [int,float]
            2/k-th spectral moment [MPa**2].
        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k: [int,float]
            Fatigue strength exponent [/].
        :return:
            Estimated damage intensity.   
        :rtype: float
        """
        d = 2**(k/2) / (2*np.pi*C) * gamma(1.0 + k/2.0) * m_2k**(k/2)
        return  d

    def get_life(self, C, k):
        """Calculate fatigue life with parameters C, k, as defined in [1,2,3].

        :param C: [int,float]
            S-N curve intercept [MPa**k].
        :param k: [int,float]
            S-N curve inverse slope [/].
        :return:
            Estimated fatigue life in seconds.
        :rtype: float
        """
        m_2k, = self.spectral_data.get_spectral_moments(self.spectral_data.PSD_splitting, moments=[2/k])[0]

        dSM = self.damage_intesity_SM(m_2k, C, k)
        T = 1.0/dSM
        return T
    
    def get_PDF(self, s):
        raise Exception(f'Function <get_PDF> is not available for class {self.__class__.__name__:s}.')