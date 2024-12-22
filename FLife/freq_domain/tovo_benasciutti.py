import numpy as np
from scipy.integrate import quad
from .narrowband import Narrowband

class TovoBenasciutti(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Tovo and Benasciutti[1, 2, 3].
      
    References
    ----------
    [1] Roberto Tovo. Cycle distribution and fatigue damage under broadband
        random loading. International Journal of Fatigue, 24(11):1137{
        1147, 2002
    [2] Denis Benasciutti and Roberto Tovo. Spectral methods for lifetime
        prediction under wide-band stationary random processes. International
        Journal of Fatigue, 27(8):867{877, 2005
    [3] Denis Benasciutti and Roberto Tovo. Comparison of spectral methods for fatigue 
        analysis of broad-band Gaussian random processes. Probabilistic Engineering Mechanics,
        21(4), 287-299, 2006
    [4] Aleš Zorman and Janko Slavič and Miha Boltežar. 
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
    >>> tb = FLife.TovoBenasciutti(sd)
    >>> print(f'Fatigue life, method 1: {tb.get_life(C,k, method="method 1"):.3e} s.')
    >>> print(f'Fatigue life, method 2: {tb.get_life(C,k, method="method 2"):.3e} s.')

    Define stress vector and depict stress peak PDF

    >>> s = np.arange(0,np.max(x),.01) 
    >>> plt.plot(s,tb.get_PDF(s, method='method 1'), lw=5, alpha=.5, label = 'method 1')
    >>> plt.plot(s,tb.get_PDF(s, method='method 2'), '--', label = 'method 2')
    >>> plt.xlabel('Stress [MPa]')
    >>> plt.ylabel('PDF')
    >>> plt.legend()
    """
    def __init__(self, spectral_data):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData
        """     
        Narrowband.__init__(self, spectral_data)

    def _calculate_coefficient(self, method='method 2'):
        """Calculate weigthing parameter b for the Tovo-Benasciutti method. Parameter b is 
            defined by Tovo and Benasciutti [1,2,3].
        
        :param method:  string
            - 'method 1': `b` weighting parameter `b` is defined by Tovo[1].
            - 'method 2': `b` weighting parameter `b` is defined by Tovo and Benasciutti [2].
                          (This is the 2005 improved method [2])
            - 'method 3': `b` weighting parameter `b` is defined by Tovo and Benasciutti [3].
                          (This is the 2006 improved method [3])
        :return b: float
        """
        if method == 'method 1': 
            b = self._calculate_coefficient_method_1()
        elif method == 'method 2': 
            b = self._calculate_coefficient_method_2()
        elif method == 'method 3': 
            b = self._calculate_coefficient_method_3()
        else: 
            raise Exception('Unrecognized Input Error')
        return b


    def _calculate_coefficient_method_1(self):
        """Calculate weigthing parameter b Tovo-Benasciutti method. Parameter b is 
            defined by Tovo[1].
        
        :return b: float
        """
        alpha1 = self.spectral_data.alpha1
        alpha2 = self.spectral_data.alpha2

        b = min( (alpha1-alpha2) / (1.0 - alpha1), 1.0)
        
        return b

    def _calculate_coefficient_method_2(self):
        """Calculate weigthing parameter b for the 2005 improved Tovo-Benasciutti method. Parameter b is 
            defined by Tovo and Benasciutti [2].
        
        :return b: float
        """
        alpha1 = self.spectral_data.alpha1
        alpha2 = self.spectral_data.alpha2

        b = (alpha1-alpha2) * (  1.112 * ( 1+ alpha1*alpha2 - (alpha1+alpha2)  ) * np.exp(2.11*alpha2) +(alpha1-alpha2) ) / ((alpha2-1)**2)

        return b
        
    def _calculate_coefficient_method_3(self):
        """Calculate weigthing parameter b for the 2006 improved Tovo-Benasciutti method. Parameter b is 
            defined by Tovo and Benasciutti [3].
        
        :return b: float
        """
        alpha075 = self.spectral_data.alpha075
        alpha2 = self.spectral_data.alpha2

        b = (alpha075**2-alpha2**2) / (1-alpha2**2)

        return b

    def get_PDF(self, s, method='method 2'):
        """Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :param method:  string

            - 'method 1': weighting parameter `b` is defined by Tovo[1].
              
            - 'method 2': weighting parameter `b` is defined by Tovo and Benasciutti [2].

            - 'method 3': weighting parameter `b` is defined by Tovo and Benasciutti [3].

        :return: function pdf(s)
        """
        alpha2 = self.spectral_data.alpha2
        m0 = self.spectral_data.moments[0]

        b = self._calculate_coefficient(method=method)

        def pdf(s):
            px = b * ((s / m0) * np.exp( - s**2 / (2 * m0))) + \
                (1 - b) * ((s / (m0 * alpha2**2)) * np.exp( - s**2 / (2 * alpha2**2 * m0))) 
            return px
        return pdf(s)

    def get_life(self, C, k, method='method 2', integrate_pdf=False):
        """Calculate fatigue life with parameters C, k, as defined in [4].

        :param C: [int,float]
            S-N curve intercept [MPa**k].
        :param k: [int,float]
            S-N curve inverse slope [/].
        :param method:  string
        
            - 'method 1': weighting parameter `b` is defined by Tovo[1].
            - 'method 2': weighting parameter `b` is defined by Tovo and Benasciutti [2].
            - 'method 3': weighting parameter `b` is defined by Tovo and Benasciutti [3].

        :param integrate_pdf:  boolean
            If true the the fatigue life is estimated by integrating the PDF, 
            Default is false which means that the theoretical equation is used
        :return:
            Estimated fatigue life in seconds.
        :rtype: float
        """ 
        if integrate_pdf:
            d = self.spectral_data.nu / C * \
                quad(lambda s: s**k*self.get_PDF(s, method=method), 
                     a=0, b=np.inf)[0]
        else:
            m0 = self.spectral_data.moments[0]
            nu = self.spectral_data.nu
            alpha2 = self.spectral_data.alpha2
            
            b = self._calculate_coefficient(method=method)

            dNB = self.damage_intesity_NB(m0=m0, nu=nu, C=C, k=k) 
            l = b + ( 1.0 - b ) * alpha2**(k-1.0)
            d = dNB * l
        
        T = 1.0/d
        return T