import numpy as np
import scipy.stats as ss
import scipy.integrate as si
import scipy.special as spe

#Check method implementation

class GaoMoan(object):
    """Class for fatigue life estimation using frequency domain 
    method by Gao and Moan [1].
       
    References
    ----------
    [1] Zhen Gao and Torgeir Moan. Frequency-domain fatigue analysis of
        wide-band stationary Gaussian processes using a trimodal spectral formulation.
        International Journal of Fatigue, 30(10-11):1944{1955, 2008
    [2] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020
    """
    def __init__(self, spectral_data):
        '''
        Constructor
        '''
        self.spectral_data = spectral_data
        
        self.psd = spectral_data.psd
        self.moments_freq = spectral_data.moments
        self.moments = spectral_data.moments  #Preveri zakaj je to (ni od omege)
        self.calculate_spectral_moments_omega = lambda psd: spectral_data.calculate_spectral_moments_frequency(psd)[:3]
        self.t = spectral_data.t
        self.split_mode = spectral_data.gao_split_mode
        
        self.modes = False #KAJ je to??
        
    def get_life(self, C, k):
        """Calculate fatigue life with parameters C, k, as defined in [2].

        :param C: [int,float]
            Fatigue strength coefficient [MPa**k].
        :param k : [int,float]
            Fatigue strength exponent [/].
        :return T: float
            Estimated fatigue life in seconds.
        """
        
        if len(self.split_mode) == 1: # narrow-band
            
            x = self.psd[:, 0]
            
            f1 = np.abs(x - self.split_mode[0]).argmin()
                        
            m = self.calculate_spectral_moments_omega(self.psd[:f1, :])
            v0 = np.sqrt(m[2] / m[0])
            
            SH = np.sqrt(2.0 * m[0])**k * spe.gamma(1.0 + k / 2.0) 
            D = self.t * v0 * SH / C
            T = self.t / D
        
            return T
            
        elif len(self.split_mode) == 2: # bi-modal
            
            x = self.psd[:,0]
            
            f1 = np.abs(x - self.split_mode[0]).argmin()
            f2 = np.abs(x - self.split_mode[1]).argmin()
            
            
            mH = self.calculate_spectral_moments_omega(self.psd[:f1, :])
            mM = self.calculate_spectral_moments_omega(self.psd[f1:f2, :])
            
            dH, dM = [np.sqrt(1.0 - mH[1]**2/(mH[0]*mH[2])), np.sqrt(1.0 - mM[1]**2/(mM[0]*mM[2]))]
            
            v0H, v0M = np.sqrt( np.array([mH[2]/mH[0], mM[2]/mM[0]]) )
            
            v0P = np.sqrt(mH[2] * dH**2 + mM[2]) * np.sqrt(mM[0]) / ( (mH[0]+mM[0]))
            
            RH = lambda x: ss.rayleigh.pdf(x / np.sqrt(mH[0])) / np.sqrt(mH[0])
            RM = lambda x: ss.rayleigh.pdf(x / np.sqrt(mM[0])) / np.sqrt(mM[0])
            
            smax = 3.0 * np.sqrt(self.moments_freq[0])
            
            dx = 0.05
            
            x = np.arange(0, smax, dx)
            
            RP = np.convolve(RH(x), RM(x)) * dx
            x2 = np.arange(0, RP.size * dx, dx)
            
            SH = np.sqrt(2.0 * mH[0])**k * spe.gamma(1.0 + k / 2.0)
            SP = si.simps(x2**k * RP, x2, dx)
            
            DH = self.t * v0H * SH / C
            DP = self.t * v0P * SP / C
            
            D = DH + DP
            T = self.t / D
                
            return T
            
        elif len(self.split_mode) == 3: # tri-modal
            self.modes = True
            
            x = self.psd[:,0]
            
            f1 = np.abs(x - self.split_mode[0]).argmin()
            f2 = np.abs(x - self.split_mode[1]).argmin()
            f3 = np.abs(x - self.split_mode[2]).argmin()
            
            self.frequencies = (f1, f2, f3)
            
            m0H, m1H, m2H, m0M, m1M, m2M, m0L, m1L, m2L = self.getModes()
            v0H, v0M, v0L = np.sqrt( np.array([m2H/m0H, m2M/m0M, m2L/m0L]) )
            
            
            dH, dM = [np.sqrt(1.0 - m1H**2/(m0H*m2H)), np.sqrt(1.0 - m1M**2/(m0M*m2M))]
           
            # -- then, get other constants for each mode
            v0P = np.sqrt(m2H * dH**2 + m2M) * np.sqrt(m0M) / ((m0H+m0M))
            v0Q = np.sqrt(m2H * dH**2 + m2M * dM**2 + m2L) * \
            (2.0 * np.sqrt(m0L * (m0H+m0M+m0L)) - np.pi * np.sqrt(m0H*m0M) + 2.0 * np.sqrt(m0H*m0M) * \
             np.arctan(np.sqrt((m0H*m0M)/m0L) / np.sqrt(m0H+m0M+m0L))) / (2.0 * np.sqrt(m0H+m0M+m0L)**3)
            
            # -- Define Rayleigh sums
            RH = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0H)) / np.sqrt(m0H)
            RM = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0M)) / np.sqrt(m0M)
            RL = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0L)) / np.sqrt(m0L)
            
            smax = 3.0 * np.sqrt(self.moments_freq[0])
            
            dx = 0.05
            x = np.arange(0, smax, dx)
    #        print smax
            
            RP = np.convolve(RH(x), RM(x)) * dx
            x2 = np.arange(0, RP.size * dx, dx)
            RQ = np.convolve(RP, RL(x2)) * dx
            x3 = np.arange(0, RQ.size*dx , dx)
            
            
            # -- Calculate damage and life
            
            # -- Define expected value of stress range ( int(S^k * p(s)) )
            SH = np.sqrt(2.0 * m0H)**k * spe.gamma(1.0 + k / 2.0)
            SP = si.simps(x2**k * RP, x2, dx)
            SQ = si.simps(x3**k * RQ, x3, dx)

            
            # -- Calculate damage D
            DH = self.t * v0H * SH / C
            DP = self.t * v0P * SP / C
            DQ = self.t * v0Q * SQ / C
            
            D = DH + DP + DQ
            T = self.t / D
            
            return T
            
        else:
            #print('Walda')
            # -- First, get moments for all 3 modes
            m0H, m1H, m2H, m0M, m1M, m2M, m0L, m1L, m2L = self.getModes()
            v0H, v0M, v0L = np.sqrt( np.array([m2H/m0H, m2M/m0M, m2L/m0L]) )#TUKI / (2*np.pi)
            #print('m0H FLife', m0H)
            #print('m2M FLife', m2M)
            dH, dM = [np.sqrt(1.0 - m1H**2/(m0H*m2H)), np.sqrt(1.0 - m1M**2/(m0M*m2M))]
           
            # -- then, get other constants for each mode
            v0P = np.sqrt(m2H * dH**2 + m2M) * np.sqrt(m0M) / ((m0H+m0M))
            v0Q = np.sqrt(m2H * dH**2 + m2M * dM**2 + m2L) * \
            (2.0 * np.sqrt(m0L * (m0H+m0M+m0L)) - np.pi * np.sqrt(m0H*m0M) + 2.0 * np.sqrt(m0H*m0M) * \
             np.arctan(np.sqrt((m0H*m0M)/m0L) / np.sqrt(m0H+m0M+m0L))) / (2.0 * np.sqrt(m0H+m0M+m0L)**3)
            # TUKI
            # -- Define Rayleigh sums
            RH = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0H)) / np.sqrt(m0H)
            RM = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0M)) / np.sqrt(m0M)
            RL = lambda x: ss.rayleigh.pdf(x/np.sqrt(m0L)) / np.sqrt(m0L)
            
            smax = 3.0 * np.sqrt(self.moments_freq[0])
            
            dx = 0.05
            x = np.arange(0, smax, dx)
    #        print smax
            
            RP = np.convolve(RH(x), RM(x)) * dx
            x2 = np.arange(0, RP.size * dx, dx)
            RQ = np.convolve(RP, RL(x2)) * dx
            x3 = np.arange(0, RQ.size*dx , dx)
            
            
            # -- Calculate damage and life
            
            # -- Define expected value of stress range ( int(S^k * p(s)) )
#                SH = (2.0 * np.sqrt(2.0 * m0H))**k * spe.gamma(1.0 + k / 2.0)
#                SP = 2.0**k * si.simps(x2**k * RP, x2, dx)
#                SQ = 2.0**k * si.simps(x3**k * RQ, x3, dx)
            
            SH = np.sqrt(2.0 * m0H)**k * spe.gamma(1.0 + k / 2.0)
            SP = si.simps(x2**k * RP, x2, dx)
            SQ = si.simps(x3**k * RQ, x3, dx)

            
            # -- Calculate damage D
            DH = self.t * v0H * SH / C
            DP = self.t * v0P * SP / C
            DQ = self.t * v0Q * SQ / C
            
            D = DH + DP + DQ
            T = self.t / D
            
            return T
            
    def getModes(self):
        '''Produce data for calculation
        based on three modes'''
        
        if not self.modes:
            frq = self.equalAreaModes()
        else:
            frq = self.frequencies
        
        mL = self.calculate_spectral_moments_omega(self.psd[:frq[0], :])
        mM = self.calculate_spectral_moments_omega(self.psd[frq[0]:frq[1], :])
        mH = self.calculate_spectral_moments_omega(self.psd[frq[1]:frq[2], :])
        
#        print mL, mM, mH
        
        return np.hstack((mH, mM, mL))
    
    def equalAreaModes(self):
        '''Divide PSD in three modes with
        equal area.'''
        y = self.psd[:,1]#[np.where(self.psd[:,0] < 1050.0)]
        q_area = np.sum(y) # (not the REAL area because x-axis unit is omitted)
        
        # -- Calculate a cumulative sum vector of y
        cy = np.cumsum(y)
        
        # -- Find where cumulative sum is equal to 1/3, 2/3 and 3/3 of A
        f1 = np.abs(cy - q_area/3.0).argmin()
        f2 = np.abs(cy - 2.0 * q_area/3.0).argmin()
        f3 = np.abs(cy - 3.0 * q_area/3.0).argmin()
        
#        x = self.psd[:,0]
#        f1 = np.abs(x - 300.0).argmin()
#        f2 = np.abs(x - 500.0).argmin()
#        f3 = np.abs(x - 700.0).argmin()
        
#        print 'Frequencies are (f1, f2, f3): (%.2f, %.2f, %.2f)' \
#        % (self.psd[:,0][f1], self.psd[:,0][f2], self.psd[:,0][f3])
        
        return (f1, f2, f3)