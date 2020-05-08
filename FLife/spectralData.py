import numpy as np
import scipy.signal as ss


class SpectralData(object):
    """SpectralData renders power spectral density, spectral moments and
    spectral band estimators, based on input signal time history. Object 
    contains values needed for subsequent fatigue life calculations. 

    Notes
    -----
    A primitive caching is used. Here is how it works: with class
    instantiation, ALL values that are expected to be used are calculated.
    The properties are created and then called. Calculation is done only at
    the beginning and only once.
    """

    def __init__(self, input, dt, window='hanning', nperseg=1280,
                 noverlap=None, psd_trim_length=None):
        """Call parent constructor, analyse input and define fatigue life
        parameters and constants.
        
        
        :param input: [array, string]
                Signal values/file-path. Input can be 1D array of values or
                path to appropriately formated .txt file.
        :param dt: float
                Sampling interval
        :param window: str or tuple or array_like, optional
                Desired window to use. Defaults to ‘hanning’.
        :param nperseg:  int, optional
                Length of each segment. Defaults to 1280.
        :param noverlap: int, optional
                Number of points to overlap between segments. If None, 
                noverlap = nperseg / 2. Defaults to None.

        Properties
        ----------
        data    : array 1D
                Signal values with time spacing dt
        dt      : float
                Time between discreete signal values    
        moments : array 1D
                Spectral moments from m0 to m4    
        psd     : array 2D
                Normalized power spectral density
        t       : float
                Length of signal in seconds, N * dt

        Raises
        ------
        ValueError, for unknown input.
        """        
        if isinstance(input, str):
            self.data = self.readf(input)
        elif isinstance(input, np.ndarray):
            self.data = input
        else:
            raise Exception('Unrecognized Input Error')

        if isinstance(dt, float):
            self.dt = dt
        else:
            raise Exception('Unrecognized Input Error')

        self.t = self.dt * self.data.size

        self.x = np.arange(0, self.t, dt)

        self.gao_split_mode = np.array([])

        self.trim_length = psd_trim_length

        self.psd = self._calculate_psd(self.data, fs=1.0/self.dt, window=window,
                                       nperseg=nperseg, noverlap=noverlap,
                                       trim=psd_trim_length)

        self.moments = self.calculate_spectral_moments_frequency(self.psd)
        self.moments_omega = self.calculate_spectral_moments_omega(self.psd)
        self._calculate_coefficients()


    def _calculate_psd(self, data, fs=1, window='hamming', nperseg=10280, noverlap=None, trim=None):
        """Calculates PSD using welch estimator."""
        f, p = ss.welch(data, fs=fs, window=window,
                        nperseg=nperseg, noverlap=noverlap)

        psd = np.vstack((f, p)).transpose()

        if trim is None:
            return psd
        else:
            df = fs / nperseg
            trim_idx = int(np.floor(trim * df))

            return psd[:trim_idx, :trim_idx]

    def _calculate_coefficients(self):
        """Calculate all of them."""
        self.m_p = self._get_mp()
        self.m075 = self._get_m075()
        self.m150 = self._get_m150()
        self.nu_p = self._get_nup()
        self.nu = self.nu_p

        self.al075 = self.m075 / (np.sqrt(self.moments[0] * self.m150))
        self.al1 = self.moments[1] / (np.sqrt(self.moments[0] * self.moments[2]))
        self.al2 = self.moments[2] / (np.sqrt(self.moments[0] * self.moments[4]))

    def calculate_spectral_moments_frequency(self, psd):
        """Calculate 0th to 4th PSD moment.
        
        Returns
        -------
        Array [m0,m1,m2,m3,m4].
        """
        f = psd[:, 0]
        p = psd[:, 1]

        m = [
            np.trapz(p, f),
            np.trapz(f*p, f),
            np.trapz(f**2 * p, f),
            np.trapz(f**3 * p, f),
            np.trapz(f**4 * p, f),
        ]

        return np.array(m)
    
    def calculate_spectral_moments_omega(self, psd):
        om = 2 * np.pi * psd[:, 0]
        p = psd[:, 1]
        m = [
            np.trapz(p, om),
            np.trapz(om*p, om),
            np.trapz(om**2 * p, om),
            np.trapz(om**3 * p, om),
            np.trapz(om**4 * p, om),
        ]

        return np.array(m)
    
    def _get_m075(self):
        '''Calculate 0.75th spectral moment.'''
        f = self.psd[:, 0]
        p = self.psd[:, 1]
        
        return np.trapz(f**0.75 * p, f)

    def _get_m150(self):
        '''Calculate 1.5th spectral moment.'''
        f = self.psd[:, 0]
        p = self.psd[:, 1]
        
        return np.trapz(f**1.5 * p, f)
    
    def _get_mp(self):
        '''Calculate M+ / nu_p'''
        return np.sqrt(self.moments[4] / self.moments[2])
    
    def _get_nup(self):
        '''Calculate nu_p - peak intensity for narrowband signal.'''
        return np.sqrt(self.moments[2] / self.moments[0])
    
#    def _get_gamma(self):
#        '''Calculate irregularity factor gamma.'''
#        return np.sqrt( 1. / \
#                           (self.moments[0] * self.moments[4])) * self.moments[2]
#
#    def TimeDomainRMS(self):
#        
#        return np.sqrt(np.sum(self.data**2.0) / self.data.size)
#    
#    def FrqDomainRMS(self):
#        
#        return np.sqrt(self.moments[0])

    def readf(self, filename):
        """Read input file and extract values in form of array (float).
        
        Values inside txt file must be separated with whitespace, e.g. space or newline.
        
        Parameters
        ----------
        filename : string
                   path to file
        
        Returns
        -------
        1D array filled with values. 
        """
        f = open(filename)
        data = np.array(map(float,f.read().split()))
        f.close()
        
        return data