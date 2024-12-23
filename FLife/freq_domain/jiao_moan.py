import numpy as np
from scipy import stats
from scipy import integrate
from scipy.special import gamma
from .narrowband import Narrowband
from ..tools import pdf_rayleigh_sum

class JiaoMoan(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Jiao and Moan [1].
    
    References
    ----------
    [1] Guoyang Jiao and Torgeir Moan. Probabilistic analysis of fatigue due to Gaussian load processes.
        Probabilistic Engineering Mechanics, 5(2):76-83, 1990
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
    >>> jm = FLife.JiaoMoan(sd, PSD_splitting=('userDefinedBands', [80,150]))
    >>> print(f'Fatigue life: {jm.get_life(C,k):.3e} s.')

    Plot segmentated PSD, used in Jiao-Moan method

    >>> lower_band_index, upper_band_index= jm.band_stop_indexes
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
        
    def get_life(self, C, k, approximation = False):
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
        if len(self.band_stop_indexes) == 1: # narrow-band
            T = self._life_NB(C,k)
        elif len(self.band_stop_indexes) == 2: # bi-modal
            T = self._life_bimodal(C,k,approximation)
        else:
            raise Exception('Specify up to bi-modal random process.')
        return T

    def _life_NB(self, C, k):
        m0H, = self.spectral_data.get_spectral_moments(self.PSD_splitting, moments=[0])[0][0]
        v0H, = self.spectral_data.get_nup(self.PSD_splitting)[0]

        # -- Define expected value of stress range ( int(S^k * p(s)) ) proces H(t), fatigue life
        dNB_H = self.damage_intesity_NB(m0=m0H, nu=v0H, C=C, k=k) 
        T = 1/dNB_H
        return T

    def _life_bimodal(self, C, k, approximation = False):
        # -- spectral moments for each narrowband
        moments = self.spectral_data.get_spectral_moments(self.PSD_splitting, moments=[0,2])
        m0L, m2L = moments[0] #spectral moments for lower band
        m0H, m2H = moments[1] #spectral moments for upper band

        # -- normalized variances
        m0 = np.sum(moments[:, 0])
        m0L_norm = m0L/m0
        m0H_norm = m0H/m0

        # -- Vanmarcke bandwidth parameter
        _, epsV_H = self.spectral_data.get_vanmarcke_parameter(self.PSD_splitting)

        # -- positive slope zero crossing frequency
        v0L, v0H = self.spectral_data.get_nup(self.PSD_splitting)
        v0P = m0L_norm * v0L* np.sqrt(1 + m0H_norm/m0L_norm * (v0H/v0L*epsV_H)**2)

        if approximation:
            # -- bandwidth correction factor
            v0 = 1/(2*np.pi) * np.sqrt((m2L + m2H)/(m0L + m0H))
            rho = v0P/v0 * (m0L_norm**(k/2 + 2) * (1 - np.sqrt(m0H_norm/m0L_norm)) \
                + np.sqrt(np.pi*m0L_norm*m0H_norm) * (k * gamma(k/2 + 1/2))/(gamma(k/2 +1))) \
                    + v0H/v0 * m0H_norm**(k/2)

            # -- damage intensity
            d_nb = self.damage_intesity_NB(m0=m0, nu=v0, C=C, k=k) 
            d = rho * d_nb
        else:
            # -- damage intensity
            dNB_H = self.damage_intesity_NB(m0=m0H, nu=v0H, C=C, k=k) 
            dNB_P = self._damage_intesity_bimodal_LF(m0_LF=m0L, m0_HF=m0H, nuP=v0P, C=C, k=k)
            d = dNB_H + dNB_P

        T = 1 / d
        return T

    def _damage_intesity_bimodal_LF(self, m0_LF, m0_HF, nuP, C, k):
        """Calculates damage intensity for low frequency component of bimodal random process,
        with parameters m0, nuP, C, k, as defined in [2].

        :param m0_LF: [int,float]
            Zeroth spectral moment of low frequency component [MPa**2].
        :param m0_HF: [int,float]
            Zeroth spectral moment of high frequency component [MPa**2].
        :param nuP: [int,float]
            Frequency of positive slope zero crossing of low frequency component[Hz].
        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return d: float
            Estimated damage intensity of low frequency component.
        """
        pdf_P = pdf_rayleigh_sum(m0_LF,m0_HF)
        S_P = integrate.quad(lambda x:  x**k * pdf_P(x), 0, np.inf)[0]
        d = nuP * S_P / C
        return d
            
    def get_PDF(self, s):
        raise Exception(f'Function <get_PDF> is not available for class {self.__class__.__name__:s}.')