import numpy as np
import scipy.stats as ss
import scipy.integrate as si
import scipy.special as spe
import warnings
from .narrowband import Narrowband

class GaoMoan(Narrowband):
    """Class for fatigue life estimation using frequency domain 
    method by Gao and Moan [1].
    
    :param spectral_data:  Instance of object SpectralData       
    :param band_frequency:  list
        List of frequencies that divides general wideband procces into 3 or less narrowband
        processes. Specified frequency is considered as upper limit of narrowband process.
        Defaults to False.

    References
    ----------
    [1] Zhen Gao and Torgeir Moan. Frequency-domain fatigue analysis of
        wide-band stationary Gaussian processes using a trimodal spectral formulation.
        International Journal of Fatigue, 30(10-11):1944{1955, 2008
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """
    def __init__(self, spectral_data, band_frequency=False):
        '''Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData       
        :param band_frequency:  list
            List of frequencies that divides general wideband procces into 3 or less narrowband
            processes. Specified frequency is considered as upper limit of narrowband process.
            Defaults to False.
         '''
        Narrowband.__init__(self, spectral_data)

        band = [] 
        if isinstance(band_frequency, list):
            for indx,freq in enumerate(band_frequency):
                if indx <3:
                    if isinstance(freq, (int,float)):
                        band.append(freq)
                    else:
                        raise Exception('Unrecognized Input Error')
                else:
                    warnings.warn("Too many frequency bands provided. Onyl first 3 will be used.")
                    break

        self.band_frequency = np.array(band)
        self.bands = False 
        
    def get_life(self, C, k):
        """Calculate fatigue life with parameters C, k, as defined in [2].

        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return T: float
            Estimated fatigue life in seconds.
        """
        
        if len(self.band_frequency) == 1: # narrow-band, user specified bandwidth

            psd = self.spectral_data.psd
            f1 = self.band_frequency[0]     
            f1_indx = np.abs(psd[:, 0]  - f1).argmin()

            # -- spectral moments, positive slope zero crossing frequency
            m0H,_,m2H,_,_ = self.spectral_data.calculate_spectral_moments_frequency(psd[:f1_indx, :])
            v0H = np.sqrt(m2H / m0H)

            # -- Define expected value of stress range ( int(S^k * p(s)) ) proces H(t), fatigue life
            dNB_H = self.damage_intesity_NB(m0=m0H, nu=v0H, C=C, k=k) 
            T = 1/dNB_H
        
            return T
            
        elif len(self.band_frequency) == 2: # bi-modal, user specified bandwidth
            
            psd = self.spectral_data.psd
            f1, f2 = self.band_frequency[0], self.band_frequency[1] 
            f1_indx = np.abs(psd[:, 0] - f1).argmin()
            f2_indx = np.abs(psd[:, 0] - f2).argmin()
            
            # -- spectral moments
            m0H,m1H,m2H,_,_ = self.spectral_data.calculate_spectral_moments_frequency(psd[:f1_indx,:])
            m0M,_,m2M,_,_ = self.spectral_data.calculate_spectral_moments_frequency(psd[f1_indx:f2_indx,:])
            
            # -- Vanmarcke bandwidth parameter
            eps_H = np.sqrt(1.0 - m1H**2/(m0H*m2H))
            
            # -- positive slope zero crossing frequency
            v0H = np.sqrt(m2H / m0H)
            v0P = np.sqrt(m2H * eps_H**2 + m2M) * (np.sqrt(m0M) /  (m0H + m0M))
            
            # -- peak pdf 
            pdf_H = lambda x: ss.rayleigh.pdf(x / np.sqrt(m0H)) / np.sqrt(m0H)
            pdf_M = lambda x: ss.rayleigh.pdf(x / np.sqrt(m0M)) / np.sqrt(m0M)
        
            # -- max stress is 3*standard deviation
            smax = 3.0 * np.sqrt(self.spectral_data.moments[0])
            dx = 0.05
            x = np.arange(0, smax, dx) 
            
            pdf_P = dx * np.convolve(pdf_H(x), pdf_M(x)) 
            x2 = np.arange(0, pdf_P.size * dx, dx)
            
            # -- Define expected value of stress range ( int(S^k * p(s)) ) P(t)
            SP = si.simps(x2**k * pdf_P, x2, dx)

            dNB_H = self.damage_intesity_NB(m0=m0H, nu=v0H, C=C, k=k) 
            dNB_P = v0P * SP / C
            
            d = dNB_H + dNB_P
            T = 1 / d
                
            return T
            
        elif len(self.band_frequency) == 3: # tri-modal, user specified band width
            self.bands = True 
            
            psd = self.spectral_data.psd
            f1, f2, f3 = self.band_frequency[0], self.band_frequency[1], self.band_frequency[2] 
            f1_indx = np.abs(psd[:, 0] - f1).argmin()
            f2_indx = np.abs(psd[:, 0] - f2).argmin()
            f3_indx = np.abs(psd[:, 0] - f3).argmin()

            self.frequencies_indx = (f1_indx, f2_indx, f3_indx)

            return self._calculate_life_trimodal(C, k)  
            
        else: # tri-modal, equal variance bandwidth
            return self._calculate_life_trimodal(C, k)  

    def _calculate_life_trimodal(self, C, k):
        '''Returns fatique life based on specified 3-modal process bandwith.
        '''
        # -- calculate equal area bands and its moments
        m0H, m1H, m2H, m0M, m1M, m2M, m0L, _, m2L = self.getBands()

        # -- positive slope zero crossing frequency
        v0H = np.sqrt(m2H/m0H)

        # -- Vanmarcke bandwidth parameter
        eps_H, eps_M = [np.sqrt(1.0 - m1H**2/(m0H*m2H)), np.sqrt(1.0 - m1M**2/(m0M*m2M))]
        
        # -- positive slope zero crossing frequency
        v0P = np.sqrt(m2H * eps_H**2 + m2M) * (np.sqrt(m0M) / (m0H + m0M))
        v0Q = np.sqrt(m2H * eps_H**2 + m2M * eps_M**2 + m2L) * \
            (2.0 * np.sqrt(m0L * (m0H + m0M + m0L)) - np.pi * np.sqrt(m0H * m0M) \
            + 2.0 * np.sqrt(m0H * m0M) * np.arctan(np.sqrt((m0H * m0M ) /m0L) / np.sqrt(m0H + m0M + m0L))) \
            / (2.0 * np.sqrt(m0H + m0M + m0L)**3)

        # -- peak pdf 
        pdf_H = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0H)) / np.sqrt(m0H)
        pdf_M = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0M)) / np.sqrt(m0M)
        pdf_L = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0L)) / np.sqrt(m0L)
        
        # -- max stress is 3*standard deviation
        smax = 3.0 * np.sqrt(self.spectral_data.moments[0])
        dx = 0.05
        x = np.arange(0, smax, dx)
        
        pdf_P = dx * np.convolve(pdf_H(x), pdf_M(x))
        x2 = np.arange(0, pdf_P.size * dx, dx)
        pdf_Q = dx * np.convolve(pdf_P, pdf_L(x2))
        x3 = np.arange(0, pdf_Q.size*dx , dx)
        
        # -- Define expected value of stress range ( int(S^k * p(s)) )
        SP = si.simps(x2**k * pdf_P, x2, dx)  #quad funkcija
        SQ = si.simps(x3**k * pdf_Q, x3, dx)

        # -- Calculate damage intensity d, fatigue life
        dNB_H = self.damage_intesity_NB(m0=m0H, nu=v0H, C=C, k=k) 
        dNB_P = v0P * SP / C
        dNB_Q = v0Q * SQ / C
        
        d = dNB_H + dNB_P + dNB_Q
        T = 1/ d
        
        return T

    def getBands(self):  
        '''Produce data for calculation
        based on three frequency bands'''
        
        if not self.bands:
            frq = self.equalAreaBands()
        else:
            frq = self.frequencies_indx
        
        psd = self.spectral_data.psd

        m_L = self.spectral_data.calculate_spectral_moments_frequency(psd[:frq[0],:])[:3] #only fist 3 moments
        m_M = self.spectral_data.calculate_spectral_moments_frequency(psd[frq[0]:frq[1],:])[:3]
        m_H = self.spectral_data.calculate_spectral_moments_frequency(psd[frq[1]:frq[2],:])[:3]
        
        return np.hstack((m_H, m_M, m_L))

    
    def equalAreaBands(self):
        '''Divide PSD in three bands with
        equal area. Bands border at frequncys f1, f2 and f3.
        '''
        psd = self.spectral_data.psd[:,1]
        q_area = np.sum(psd) # (not the REAL area because x-axis unit is omitted)
        
        # -- Calculate a cumulative sum vector of psd
        c_psd = np.cumsum(psd)
        
        # -- Find where cumulative sum is equal to 1/3, 2/3 and 3/3 of A
        f1 = np.abs(c_psd - q_area/3.0).argmin()
        f2 = np.abs(c_psd - 2.0 * q_area/3.0).argmin()
        f3 = np.abs(c_psd - 3.0 * q_area/3.0).argmin()
         
        return (f1, f2, f3)