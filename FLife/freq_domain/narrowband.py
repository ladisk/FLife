import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

class Narrowband(object):
    """Class for fatigue life estimation using frequency domain 
    method by Miles [1] / Bendat and Piersol [2].
   
    References
    ----------
    [1] John W. Miles. On structural fatigue under random loading. Journal
        of the Aeronautical Sciences, 21(11):753{762, 1954.
    [2] Julius S. Bendat and Allen G. Piersol. Measurement and Analysis of Random Data.
        Wiley, 1966.
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
    >>> PSD = es.get_psd(freq, 20, 60, variance = 5)  # one-sided flat-shaped PSD

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
    >>> nb = FLife.Narrowband(sd)
    >>> print(f'Fatigue life: {nb.get_life(C,k):.3e} s.')

    Define stress vector and depict stress peak PDF

    >>> s = np.arange(0,np.max(x),.01)
    >>> plt.plot(s,nb.get_PDF(s))
    >>> plt.xlabel('Stress [MPa]')
    >>> plt.ylabel('PDF')
    """
    def __init__(self, spectral_data):
        """Get needed values from reference object.
        
        :param spectral_data:  Instance of class SpectralData
        """                        
        self.spectral_data = spectral_data

    def get_PDF(self,s):
        """Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :return: function pdf(s)
        """
        m0 = self.spectral_data.moments[0]

        def pdf(s):
            px = (s / m0) * np.exp( - s**2 / (2 * m0)) 
            return px
        return pdf(s)

    def damage_intesity_NB(self, m0, nu, C, k):
        """Calculates narrowband damage intensity with parameters m0, nu, C, k, as defined in [2].

        :param m0: [int,float]
            Zeroth spectral moment [MPa**2].
        :param nu: [int,float]
            Frequency of positive slope zero crossing [Hz].
        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k: [int,float]
            Fatigue strength exponent [/].
        :return:
            Estimated damage intensity.   
        :rtype: float
        """
        d = nu * np.sqrt(2 * m0)**k * gamma(1.0 + k/2.0) / C  
        return  d

    def get_life(self, C, k, integrate_pdf=False):
        """Calculate fatigue life with parameters C, k, as defined in [2].

        :param C: [int,float]
            S-N curve intercept [MPa**k].
        :param k: [int,float]
            S-N curve inverse slope [/].
        :param integrate_pdf:  boolean
            If true the the fatigue life is estimated by integrating the PDF, 
            Default is false which means that the theoretical equation is used
        :return:
            Estimated fatigue life in seconds.
        :rtype: float
        """ 
        if integrate_pdf:
            d = self.spectral_data.nu / C * \
                quad(lambda s: s**k*self.get_PDF(s), 
                     a=0, b=np.inf)[0]
        else:
            m0 = self.spectral_data.moments[0]
            nu = self.spectral_data.nu
            d = self.damage_intesity_NB(m0=m0, nu=nu, C=C, k=k) 

        T = 1.0/d
        return T