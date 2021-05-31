import numpy as np
from scipy import stats
from scipy.integrate import quad
import warnings

class Rice(object):
    """Deprecated - Class for fatigue life estimation using frequency domain 
    method by Rice[1].

    References
    ----------
    [1] Stephen O. Rice. Mathematical analysis of random noise. The Bell
        System Technical Journal, 24(1):46-156, 1945.
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
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

    Define S-N curve parameters and get fatigue-life estimatate

    >>> C = 1.8e+22  # S-N curve intercept [MPa**k]
    >>> k = 7.3 # S-N curve inverse slope [/]
    >>> rice = FLife.Rice(sd)
    >>> print(f'Fatigue life: {rice.get_life(C,k):.3e} s.')

    Define stress vector and depict stress peak PDF
    
    >>> s = np.arange(-np.max(x),np.max(x),.01) #also negative stress
    >>> plt.plot(s,rice.get_PDF(s))
    >>> plt.xlabel('Stress [MPa]')
    >>> plt.ylabel('PDF')
    """
    def __init__(self, spectral_data):
        """Get needed values from reference object.

        :param spectral_data: Instance of class SpectralData
        """                
        self.spectral_data = spectral_data
        warnings.warn('Rice method has been deprecated since version 1.2 and will be removed in the future. ' + 
        'Peak amplitude probability density function <get_PDF> is renamed to <get_peak_PDF> and moved to class SpectralData.')

    def get_PDF(self, s):
        """Returns peak amplitude PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :return: function pdf(s)
        """
        m0 = self.spectral_data.moments[0]
        alpha2 = self.spectral_data.alpha2

        def pdf(s):
            px = np.sqrt(1.0 - alpha2**2)/np.sqrt(2.0 * np.pi * m0) * \
                np.exp( - (s**2) / (2.0 * m0 * (1.0 - alpha2**2))) +\
                alpha2*s/m0 * np.exp( - (s**2) / (2*m0)) * \
                stats.norm.cdf((alpha2 * s) / (np.sqrt(m0 * (1 - alpha2**2))))
            return px
        return pdf(s)

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
        d = self.spectral_data.m_p / C * \
            quad(lambda s: s**k*self.get_PDF(s), 
                    a=0, b=np.Inf)[0]

        T = 1.0/d

        return T