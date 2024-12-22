import numpy as np
from scipy import special
from scipy.integrate import quad

class Dirlik(object):
    """Class for fatigue life estimation using frequency domain method by Dirlik [1].
   
    References
    ----------
    [1] Turan Dirlik. Application of computers in fatigue analysis. PhD thesis,
        University of Warwick, 1985
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
    >>> dirlik = FLife.Dirlik(sd) # Dirlik's fatigue-life estimator
    >>> print(f'Dirlik: {dirlik.get_life(C,k):.3e} s.')   

    Define stress vector and depict peak stress PDF

    >>> s = np.arange(0,np.max(x),.01)
    >>> plt.plot(s,dirlik.get_PDF(s))
    >>> plt.xlabel('Stress [MPa]')
    >>> plt.ylabel('PDF')
    """
    def __init__(self, spectral_data):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData
        """          
        self.spectral_data = spectral_data
        self._calculate_coefficients() 

    def _calculate_coefficients(self):
        """Calculate coefficients for Dirlik method [2].
        """
        m0,m1,m2, _, m4 = self.spectral_data.moments
        
        c = np.empty(8)
        c[0] = ( 1. / np.sqrt(m0) ) 
        c[1] = ( ( m1 / m0 ) * np.sqrt( m2 / m4 ) ) 
        c[2] = ( np.sqrt( 1. / ( m0 * m4 ) ) * m2 ) 
        c[3] = ( 2. * ( c[1] - c[2]**2 ) / ( 1. + c[2]**2 ) ) 
        c[4] = ( ( c[2] - c[1] - c[3]**2 ) / (1 - c[2] - c[3] + c[3]**2 ) ) 
        c[5] = ( ( 1. - c[2] - c[3] + c[3]**2 ) / ( 1. - c[4] ) ) 
        c[6] = ( ( 1. - c[3] - c[5] ) ) 
        c[7] = ( 1.25 * ( c[2] - c[6] - c[5] * c[4] ) / c[3] ) 
        
        self.coeff = c
            
    def get_PDF(self,s):
        """Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector
        :return: function pdf(s)
        """
        m0 = self.spectral_data.moments[0]
        Z1 = self.coeff[0]
        R = self.coeff[4]
        Q = self.coeff[7]
        G1 = self.coeff[3]
        G2 = self.coeff[5]
        G3 = self.coeff[6]

        def pdf(s):
            Z =  Z1*s
            px = (1.0/np.sqrt(m0)) * ( \
                            (G1/Q)*np.exp(-Z/Q) + \
                            ((G2*Z)/(R**2))*np.exp(-((Z)**2)/(2.*R**2)) + \
                            (G3*Z)*np.exp(-((Z)**2)/2.) \
                            )
            return px
        return pdf(s)

    def get_life(self, C, k, integrate_pdf=False):
        """Calculate fatigue life with parameters C, k, as defined in [2].

        :param C: [int,float]
            S-N curve intercept [MPa**k].
        :param k: [int,float]
            S-N curve inverse slope [/].
        :param integrate_pdf:  boolean
            If true the the fatigue life is estimated by integrating the PDF. Default is false which means that the theoretical equation is used. 
        :return: 
            Estimated fatigue life in seconds.
        :rtype: float
        """ 
        if integrate_pdf:
            d = self.spectral_data.m_p / C * \
                quad(lambda s: s**k*self.get_PDF(s), 
                     a=0, b=np.inf)[0]
        else:
            m0 = self.spectral_data.moments[0]
            
            R = self.coeff[4]
            Q = self.coeff[7]
            G1 = self.coeff[3]
            G2 = self.coeff[5]
            G3 = self.coeff[6]
            
            d = 1 / C * ( self.spectral_data.m_p 
                          * np.sqrt(m0)**k 
                          * (
                                G1 * (Q**k)*special.gamma(1.0+k)+(np.sqrt(2.0)**k)*\
                                special.gamma(1.+k/2.)*(G2 * abs(R)**k+G3)\
                            )
                        )
    
        T = 1/d
        return T