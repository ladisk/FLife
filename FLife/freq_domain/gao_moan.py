import numpy as np
from scipy import stats
from scipy import integrate
from .jiao_moan import JiaoMoan
from ..tools import pdf_rayleigh_sum

class GaoMoan(JiaoMoan):
    """Class for fatigue life estimation using frequency domain 
    method by Gao and Moan [1].
    
    References
    ----------
    [1] Zhen Gao and Torgeir Moan. Frequency-domain fatigue analysis of
        wide-band stationary Gaussian processes using a trimodal spectral formulation.
        International Journal of Fatigue, 30(10-11): 1944-1955, 2008
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
    >>> PSD_middle = es.get_psd(freq, 100, 120, variance = 1)  # middle mode of random process
    >>> PSD_higher = es.get_psd(freq, 300, 350, variance = 2)  # higher mode of random process
    >>> PSD = PSD_lower + PSD_middle + PSDb_higher # trimodal one-sided flat-shaped PSD

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
    >>> gm = FLife.GaoMoan(sd, PSD_splitting=('userDefinedBands', [80,150,400]))  # fatigue-life estimator
    >>> print(f'Fatigue life: {gm.get_life(C,k):.3e} s.')   

    Plot segmentated PSD, used in Gao-Moan method

    >>> lower_band_index, middle_band_index, upper_band_index= gm.band_stop_indexes
    >>> plt.plot(sd.psd[:,0], sd.psd[:,1])
    >>> plt.vlines(sd.psd[:,0][lower_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> # lower band
    >>> plt.fill_between(sd.psd[:lower_band_index,0], sd.psd[:lower_band_index,1], 'o', label='lower band', alpha=.2, color='blue')
    >>> plt.vlines(sd.psd[:,0][middle_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> # middle band
    >>> plt.fill_between(sd.psd[lower_band_index:middle_band_index,0], sd.psd[lower_band_index:middle_band_index,1], 'o', label='middle band', alpha=.5, color ='orange')
    >>> plt.vlines(sd.psd[:,0][upper_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> # upper band
    >>> plt.fill_between(sd.psd[middle_band_index:upper_band_index,0], sd.psd[middle_band_index:upper_band_index,1], 'o', label='upper band', alpha=.5, color ='green')
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('PSD')
    >>> plt.xlim(0,450)
    >>> plt.legend()
    """
    def __init__(self, spectral_data, PSD_splitting = ('equalAreaBands', 3)):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData       
        :param PSD_splitting: tuple
                PSD_splitting[0] is PSD spliting method, PSD_splitting[1] is method argument. 
                Splitting methods:

                - 'userDefinedBands', PSD_splitting[1] must be of type list or tupple, with N 
                  elements specifying upper band frequencies of N random processes.
                - 'equalAreaBands', PSD_splitting[1] must be of type int, specifying N random processes.

                Defaults to ('equalAreaBands', 3).
        """
        JiaoMoan.__init__(self, spectral_data, PSD_splitting)
        
    def get_life(self, C, k):
        """Calculate fatigue life with parameters C, k, as defined in [1, 2].

        :param C: [int,float];
            S-N curve intercept [MPa**k].
        :param k: [int,float];
            S-N curve inverse slope [/].
        :return:
            Estimated fatigue life in seconds.
        :rtype: float
        """
        if len(self.band_stop_indexes) == 1: # narrow-band
            T = self._life_NB(C,k)
        elif len(self.band_stop_indexes) == 2: # bi-modal
            T = self._life_bimodal(C,k)
        elif len(self.band_stop_indexes) == 3: # tri-modal
            T = self._life_trimodal(C,k)
        else:
            raise Exception('Specify up to tri-modal random process.')
        return T

    def _life_bimodal(self, C, k):
        # -- spectral moments for each narrowband
        moments = self.spectral_data.get_spectral_moments(self.PSD_splitting, moments=[0,2])
        m0L, m2L = moments[0] #spectral moments for lower band
        m0H, m2H = moments[1] #spectral moments for upper band

        # -- Vanmarcke bandwidth parameter
        _, epsV_H = self.spectral_data.get_vanmarcke_parameter(self.PSD_splitting)

        # -- positive slope zero crossing frequency
        _, v0H = self.spectral_data.get_nup(self.PSD_splitting)
        v0P = 1/(2*np.pi) * (np.sqrt(m0L) / (m0H + m0L)) * np.sqrt(m2H * epsV_H**2 + m2L) 
        #v0P1 = m0L_norm * v0L* np.sqrt(1 + m0H_norm/m0L_norm * (v0H/v0L*epsV_H)**2) #jiao-moan, izraz je ekvivalenten za bimodalen proces

        # -- damage intensity
        dNB_H = self.damage_intesity_NB(m0=m0H, nu=v0H, C=C, k=k) 
        dNB_P = self._damage_intesity_bimodal_LF(m0L=m0L, m0H=m0H, nuP=v0P, C=C, k=k)  #deduje od jiao-moana

        d = dNB_H + dNB_P
        T = 1 / d
        return T

    def _life_trimodal(self, C, k):
        moments = self.spectral_data.get_spectral_moments(self.PSD_splitting, moments=[0,2])
        m0L, m2L = moments[0]
        m0M, m2M = moments[1]
        m0H, m2H = moments[2]

        # -- Vanmarcke bandwidth parameter
        _, epsV_M, epsV_H = self.spectral_data.get_vanmarcke_parameter(self.PSD_splitting)

        # -- positive slope zero crossing frequency
        _, _, v0H = self.spectral_data.get_nup(self.PSD_splitting)

        # -- positive slope zero crossing frequency
        # -- process HF + MF
        v0P = 1/(2*np.pi) * (np.sqrt(m0M) / (m0H + m0M)) * np.sqrt(m2H * epsV_H**2 + m2M) 

        # -- process HF + MF + LF
        v0Q = 1/(4*np.pi) * np.sqrt(m2H * epsV_H**2 + m2M * epsV_M**2 + m2L) * \
            (2.0 * np.sqrt(m0L * (m0H + m0M + m0L)) - np.pi * np.sqrt(m0H * m0M) \
            + 2.0 * np.sqrt(m0H * m0M) * np.arctan(np.sqrt((m0H * m0M ) /m0L) / np.sqrt(m0H + m0M + m0L))) \
            / (np.sqrt(m0H + m0M + m0L)**3)

        # -- damage intensity
        dNB_H = self.damage_intesity_NB(m0=m0H, nu=v0H, C=C, k=k) 
        dNB_P = self._damage_intesity_bimodal_LF(m0_LF=m0M, m0_HF=m0H, nuP=v0P, C=C, k=k)
        dNB_Q = self._damage_intesity_trimodal_LF(m0_LF=m0L,m0_MF=m0M, m0_HF=m0H, nu_L=v0Q, C=C, k=k)

        d = dNB_H + dNB_P + dNB_Q
        T = 1 / d
        return T

    def _damage_intesity_trimodal_LF(self, m0_LF, m0_MF, m0_HF, nu_L, C, k):
        """Calculates damage intensity for low frequency component of bimodal random process,
        with parameters m0, nuP, C, k, as defined in [2].
        :param m0_LF: [int,float]
            Zeroth spectral moment of low-frequency component [MPa**2].
        :param m0_MF: [int,float]
            Zeroth spectral moment of medium-frequency component [MPa**2].
        :param m0_HF: [int,float]
            Zeroth spectral moment of high-frequency component [MPa**2].    
        :param nu_L: [int,float]
            Frequency of positive slope zero crossing of low frequency component[Hz].
        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return d: float
            Estimated damage intensity of low frequency component.
        """
        # medium-amplitude cycle pdf
        pdf_M = pdf_rayleigh_sum(m0_MF, m0_HF)
        # LF component pdf - rayleigh distributed
        pdf_LF = lambda s: stats.rayleigh.pdf(s/np.sqrt(m0_LF)) / np.sqrt(m0_LF)
        # large-amplitude cycle pdf
        pdf_L = lambda s: np.convolve(pdf_M(s), pdf_LF(s))[0] # large cycle pdf

        S_L = integrate.quad(lambda s: s**k * pdf_L(s), 0, np.inf)[0] 
        d = nu_L * S_L / C
        return d