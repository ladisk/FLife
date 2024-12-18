import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
import warnings

class JunPark(object):
    """Class for fatigue life estimation using frequency domain 
    method by Jun and Park[1].
    
    References
    ----------
    [1] Seock-Hee Jun and Jun-Bum Park. Development of a novel fatigue damage model for 
        Gaussian wide band stress responses using numerical approximation methods. International 
        Journal of Naval Architecture and Ocean Engineering, 12: 755-767, 2020
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
    >>> jp = FLife.Jun(sd)
    >>> print(f'Fatigue life: {jp.get_life(C,k):.3e} s.')

    Define stress vector and depict stress peak PDF

    >>> s = np.arange(0,np.max(x),.01) 
    >>> plt.plot(s,jp.get_PDF(s))
    >>> plt.xlabel('Stress [MPa]')
    >>> plt.ylabel('PDF')
    """
    def __init__(self, spectral_data):
        """Get needed values from reference object.

        :param spectral_data:  Instance of class SpectralData
        """     
        self.spectral_data = spectral_data
        self._set_distribution_parameters() #calculate distribution parameters

    def get_PDF(self, s):
        """Returns cycle PDF(Probability Density Function) as a function of stress s.

        :param s:  numpy.ndarray
            Stress vector.
        :return: function pdf(s)
        """
        m0 = self.spectral_data.moments[0]
        Qc = self._get_Qc() # PDF correction factor Qc

        def jun_pdf(s):
            #PDF of stress amplitude normalized by standard deviation of process
            #exponential
            exponential_pdf = lambda s: 1/self.parameters['sigma_E'] * np.exp(-s/self.parameters['sigma_E'])
            #Rayleigh
            rayleigh1_pdf = lambda s: s/self.parameters['sigma_R']**2 * np.exp(-s**2/(2*self.parameters['sigma_R']**2))
            #Rayleigh with unit variance
            rayleigh2_pdf = lambda s: s * np.exp(-s**2/2)
            #half-Gaussian
            gauss_pdf = lambda s: 2/(np.sqrt(2*np.pi)*self.parameters['sigma_H'])* np.exp(-s**2/(2*self.parameters['sigma_H']**2))

            pdf = self.parameters['D_1']*exponential_pdf(s) + self.parameters['D_2']*rayleigh1_pdf(s) \
                + self.parameters['D_3']*rayleigh2_pdf(s) + self.parameters['D_4']*gauss_pdf(s)
            return  Qc * pdf
        
        return 1/np.sqrt(m0) * jun_pdf(s/np.sqrt(m0))

    def get_life(self, C, k, integrate_pdf=False):
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
        m_p = self.spectral_data.m_p

        if integrate_pdf:
            d = m_p / C * quad(lambda s: s**k*self.get_PDF(s), a=0, b=np.inf)[0]
        else:
            Qc = self._get_Qc()
            d = Qc * m_p / C * (np.sqrt(2*m0))**k * (self.parameters['D_1']/(np.sqrt(2)**k)* self.parameters['sigma_E']**k * gamma(1+k) \
                + self.parameters['D_2']*self.parameters['sigma_R']**k * gamma(1 + k/2) + self.parameters['D_3']*gamma(1+k/2) \
                + self.parameters['D_4']/(np.sqrt(np.pi)) * self.parameters['sigma_H']**k * gamma((1+k)/2) )

        T = 1.0/d
        return T

    def _set_distribution_parameters(self):
        '''Define PDF parameters; 
        '''
        #Parameters for n-th moment of rainflow range distrubution Mrr(n)
        alpha_1 = self.spectral_data.alpha1
        alpha_2 = self.spectral_data.alpha2
        rho = alpha_1**1.1 * alpha_2**0.9

        # Define special bandwidth parameter mu_k [1]
        def get_mu_k(k):
            nominator = self.spectral_data.get_spectral_moments(self.spectral_data.PSD_splitting, moments=[k+0.01])[0][0]
            denominator = np.sqrt(self.spectral_data.get_spectral_moments(self.spectral_data.PSD_splitting, moments=[0.01])[0][0]\
                 * self.spectral_data.get_spectral_moments(self.spectral_data.PSD_splitting, moments=[2*k+0.01])[0][0])
            return nominator/denominator

        mu_1 = get_mu_k(1)
        mu_0_52 = get_mu_k(0.52)

        #Mrr(n)
        MRR_1 = rho * mu_1**-0.96
        MRR_2 = rho * mu_1**-0.02
        MRR_3 = rho * mu_0_52

        #distribution parameters
        sigma_R = alpha_2
        D_1 = 2*(alpha_1*alpha_2 - alpha_2**2)/(1 + alpha_2**2)
        D_2 = (MRR_2 - MRR_3)/(sigma_R**2*(1 - sigma_R))
        D_3 = (-sigma_R*MRR_2 + MRR_3)/(1 - sigma_R)
        D_4 = 1 - D_1 - D_2 - D_3

        A_1 = gamma(2)/(np.sqrt(2) * gamma(1.5))
        B_1 = 1/np.sqrt(np.pi) * gamma(1)/gamma(1.5)

        sigma_H = 1/(B_1 * D_4) * (MRR_1 - D_1**2 - D_2*sigma_R - D_3)
        sigma_E = 1/(A_1 * D_1) * (MRR_1 - D_2*sigma_R - D_3 - B_1*D_4*sigma_H)

        self.parameters = {}
        self.parameters['D_1'] = D_1
        self.parameters['D_2'] = D_2
        self.parameters['D_3'] = D_3
        self.parameters['D_4'] = D_4
        self.parameters['sigma_R'] = sigma_R
        self.parameters['sigma_H'] = sigma_H
        self.parameters['sigma_E'] = sigma_E

    def _get_Qc(self):
        '''Define PDF correction factor Qc[1]; 
        '''
        alpha_1 = self.spectral_data.alpha1
        alpha_2 = self.spectral_data.alpha2

        # Correction factor Qc is validated under following conditions
        if not 0 <= alpha_1 - alpha_2 <= 1 and 0 <= alpha_2 <= 1 and np.sqrt(1-alpha_1**2) > 0.3:
            warnings.warn('Correction factor Qc is not validated for given alpha_1 and alpha_2. Results should be evaluated carefully.')

        delta_alpha = alpha_1 - alpha_2
        Qc = 0.903 -0.28*delta_alpha + 4.448*delta_alpha**2 - 15.739*delta_alpha**3 + 19.57*delta_alpha**4 \
            -8.054*delta_alpha**5 + 1.013*alpha_2 - 4.178*alpha_2**2 + 8.362*alpha_2**3 - 7.993*alpha_2**4 + 2.886*alpha_2**5

        return Qc