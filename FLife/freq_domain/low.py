from ast import Raise
import numpy as np
from scipy import stats
from scipy import integrate
from scipy import special
import warnings

class Low(object):
    """Class for fatigue life estimation using frequency domain 
    method by Low[1].

    Notes
    -----
    Numerical implementation supports only integer values of  
    S-N curve parameter k (inverse slope). Due to approximation of
    large stress cycles through McLaurin series, sufficient engineering
    precision is up to k=6 [1].

    References
    ----------
    [1] Y.M.Low. A method for accurate estimation of the fatigue damage 
        induced by bimodal processes. Probabilistic Engineering Mechanics, 25(1):75-85, 2010
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
    >>> low = FLife.Low(sd, PSD_splitting=('userDefinedBands', [80,150]))
    >>> print(f'Fatigue life: {low.get_life(C,int(k)):.3e} s.')

    Plot segmentated PSD, used in Low's method

    >>> lower_band_index, upper_band_index= low.band_stop_indexes
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
        self.spectral_data = spectral_data
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
        #Data type check
        if not isinstance(k, int):
            raise Exception('Parameter `k` must be integer. Sufficient engineering precision is up to k=6.')
        
        if k > 6:
            warnings.warn(f'Sufficient engineering precision is up to k=6. Results should be evaluated carefully.')
        
        # -- spectral moments for each narrowband
        moments = self.spectral_data.get_spectral_moments(self.PSD_splitting, moments=[0])
        m0L, = moments[0] #spectral moments for lower band
        m0H, = moments[1] #spectral moments for upper band
        
        # -- positive slope zero crossing frequency
        v0L, v0H = self.spectral_data.get_nup(self.PSD_splitting)
        v0Small = v0H - v0L  #frequency of small cycless

        #band's zero crossing frequency ratio
        beta = v0H/v0L

        #Damage from small cycles
        #stress discrepancy
        eps = lambda r_lf, phi, beta: np.pi / (2*beta) * r_lf * np.abs(np.sin(phi))
        #peak PDF
        pdf_r = lambda r, var: stats.rayleigh.pdf(r, scale=np.sqrt(var))
        #phase angle
        pdf_phi = lambda phi, beta: stats.uniform.pdf(phi, loc=np.pi/4/beta, scale=np.pi/2-np.pi/4/beta)

        #damage
        int_func_small = lambda phi, r_lf: self._inner_integral_small(k, eps(r_lf, phi, beta), m0H) * pdf_phi(phi, beta) * pdf_r(r_lf, m0L)
        I_small = integrate.dblquad(int_func_small, 0, np.inf, lambda r_lf: np.pi/4/beta, lambda r_lf: np.pi/2)[0]
        d_small = v0Small / C * I_small

        #Damage from large cycles
        int_func_large = lambda r_hf, r_lf: self._inner_integral_large(k, r_lf, r_hf, beta) *  pdf_r(r_hf, m0H) * pdf_r(r_lf, m0L)
        I_large = 1/np.pi * integrate.dblquad(int_func_large, 0, np.inf, lambda r_lf: 0, lambda r_lf: np.inf)[0]
        d_large = v0L / C  * I_large 

        #Agregated damage
        d = d_small + d_large
        T = 1 / d
        return T

    def _Ik(self, eps, var, K):
        """Calculates coefficients for binomial series expansion for small cycles damage estimation[1].
        """
        Ik_array = np.zeros(K)
        Ik_array[0] = np.exp(-eps**2/(2*var))
        Ik_array[1] = eps * Ik_array[0] + np.sqrt(2*np.pi) * np.sqrt(var) * stats.norm.cdf(-eps/np.sqrt(var))
        for i in range(K-2):
            term1 = eps**(i+2) * Ik_array[0]
            term2 = (i+2) * var * Ik_array[i]
            Ik_array[i+2] = term1 + term2
        return Ik_array

    def _inner_integral_small(self, k, eps, var):
        """Returns analytical expression of innermost integral for small cycles damage estimation,
        based on binomial series expansion[1].
        """
        out = 0
        Ik_array = self._Ik(eps,var,int(k)+1)
        for i in range(len(Ik_array)):
            out += special.binom(k,i) * (-eps)**(k-i) * Ik_array[i]
        return out

    def _ro_j(self, r_lf, r_hf, beta, j):
        """Calculates coefficients for MacLaurin series expansion for large cycles damage estimation[1].
        """
        c = beta * r_hf / (r_lf + beta**2 * r_hf)
        out = r_lf * c**j + r_hf * (beta*c -1)**j
        out /= special.factorial(j)
        return out

    def _inner_integral_large(self, k, r_lf, r_hf, beta):
        """Returns analytical approximation of innermost integral for large cycles damage estimation,
        based on MacLaurin series expansion. The approximation gives sufficient engineering
        precision in the damage estimate for up to k = 6 [1].
        """
        r_sum = r_lf + r_hf
        ro_2 = self._ro_j(r_lf, r_hf, beta, 2)
        ro_4 = self._ro_j(r_lf, r_hf, beta, 4)
        ro_6 = self._ro_j(r_lf, r_hf, beta, 6)
        
        out = r_sum**k * (np.pi - 1/3/r_sum * k * ro_2 * np.pi**3 \
            + k/5/r_sum * (ro_4 + (k-1) * ro_2**2 /2/r_sum) * np.pi**5 \
            - k/7/r_sum * (ro_6 + (k-1) * ro_2 * ro_4 /r_sum + (k-1) * (k-2) * ro_2**3 /(6*r_sum**2)) * np.pi**7)
        return out