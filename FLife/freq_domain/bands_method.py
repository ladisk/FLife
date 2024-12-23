import numpy as np
from .narrowband import Narrowband

class BandsMethod(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Braccesi et al. [1].
    
    References
    ----------
    [1] Claudio Braccesi, Filippo Cianetti and Lorenzo Tomassini. Random fatigue. A new 
        frequency domain criterion for the damage evaluation of mechanical components.
        International Journal of Fatigue, 70:417-427, 2015
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
    >>> bm = FLife.BandsMethod(sd, PSD_splitting=('userDefinedBands', [80,150]))
    >>> print(f'Fatigue life: {bm.get_life(C,k):.3e} s.')

    Plot segmentated PSD, used in Bands method

    >>> lower_band_index, upper_band_index= bm.band_stop_indexes
    >>> plt.plot(sd.psd[:,0], sd.psd[:,1])
    >>> plt.vlines(sd.psd[:,0][lower_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> plt.fill_between(sd.psd[:lower_band_index,0], sd.psd[:lower_band_index,1], 'o', label='lower band', alpha=.2, color='blue')
    >>> plt.vlines(sd.psd[:,0][upper_band_index], 0, np.max(sd.psd[:,1]), 'k', linestyles='dashed', alpha=.5)
    >>> plt.fill_between(sd.psd[lower_band_index:upper_band_index,0], sd.psd[lower_band_index:upper_band_index,1], 'o', label='upper band', alpha=.5, color ='orange')
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('PSD')
    >>> plt.xlim(0,200)
    >>> plt.legend()
    """
    def __init__(self, spectral_data, PSD_splitting = ('equalAreaBands', 1)):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData       
        :param PSD_splitting: tuple
                PSD_splitting[0] is PSD spliting method, PSD_splitting[1] is method argument. 
                Splitting methods:

                - 'userDefinedBands', PSD_splitting[1] must be of type list or tupple, with N 
                  elements specifying upper band frequencies of N random processes.
                - 'equalAreaBands', PSD_splitting[1] must be of type int, specifying N random processes.

                Defaults to ('equalAreaBands', 1).
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
        :return:
            Estimated fatigue life in seconds.
        :rtype: float
        """
        #Reference frequency (is arbitrary and is taken as expected frequency of positive 
        #slope zero crossing of random process (described by SpectralData instance))
        f_ref, = self.spectral_data.get_nup(self.spectral_data.PSD_splitting) 

        if len(self.band_stop_indexes) == 1: # modified PSD formulation
            m0_ref = self._m0_ref_modifiedPSD(k,f_ref)
        else: # user defined bands
            m0_ref = self._m0_ref_userDefinedBands(k,f_ref)

        d = self.damage_intesity_NB(m0=m0_ref, nu=f_ref, C=C, k=k) 
        T = 1/d     
        return T


    def _m0_ref_userDefinedBands(self, k, f_ref):
        # expected frequency of positive slope zero crossing
        bands_central_freq = self.spectral_data.get_nup(self.PSD_splitting)
        # bands variance
        m0_array = self.spectral_data.get_spectral_moments(self.PSD_splitting, moments=[0])[:,0]

        f = self.spectral_data.psd[:, 0]
        m0_ref_list = []
        for m0, band_freq in zip(m0_array, bands_central_freq):
            band_freq = f[np.abs(f - band_freq).argmin()]
            mo_ref = (band_freq / f_ref)**(2/k) * m0
            m0_ref_list.append(mo_ref)
        m0_ref_list = np.array(m0_ref_list)

        m0_ref = np.sum(m0_ref_list)
        return m0_ref

    def _m0_ref_modifiedPSD(self, k, f_ref):
        f = self.spectral_data.psd[:, 0]
        PSD = self.spectral_data.psd[:, 1] 
        modified_PSD = []
        df = f[1] - f[0] 

        for f_i, PSD_i in zip(f,PSD):
            PSD_i_modified = (f_i / f_ref)**(2/k) * PSD_i
            modified_PSD.append(PSD_i_modified)
        modified_PSD = np.array(modified_PSD)

        if np.__version__>='2.0.0':
            trapezoid = np.trapezoid
        else:
            trapezoid = np.trapz
            
        m0_ref = trapezoid(modified_PSD, dx=df)
        return m0_ref

    def get_PDF(self, s):
        raise Exception(f'Function <get_PDF> is not available for class {self.__class__.__name__:s}.')