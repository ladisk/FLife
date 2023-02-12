import numpy as np
import fatpack
import rainflow

class Rainflow(object):
    """Class for fatigue life estimation using rainflow counting method [1, 2].

    References
    ----------
    [1] C. Amzallag et. al. Standardization of the rainflow counting method for
        fatigue analysis. International Journal of Fatigue, 16 (1994) 287-293

    [2] ASTM E1049-85

    [3] Janko Slavič, Matjaž Mršnik, Martin Česnik, Jaka Javh, Miha Boltežar. 
        Vibration Fatigue by Spectral Methods, From Structural Dynamics to Fatigue Damage
        – Theory and Experiments, ISBN: 9780128221907, Elsevier, 1st September 2020

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
    >>> rf = FLife.Rainflow(sd)
    >>> print(f'Fatigue life: {rf.get_life(C,k):.3e} s.')
    """
    def __init__(self, spectral_data, **kwargs):
        """Get needed values from reference object.

        :param spectral_data:  Instance of object SpectralData
        :param rg: Instance of numpy.random._generator.Generator
            Parameter `rg` cotrols phase of generated time history, if not already exist in spectral_data.
        """          
        self.spectral_data = spectral_data

        #set time history if not exsist in spectral_data
        if not hasattr(spectral_data, 'data'):
            rg = kwargs.get('rg', None)
            
            f = spectral_data.psd[:,0]
            psd = spectral_data.psd[:,1]
            
            # se parametra T in fs
            if 'T' and 'fs' in kwargs.keys():             
                self.spectral_data._set_time_history(f=f, psd=psd, **kwargs)
            else:
                raise Exception('Time history is not set; T and fs must be specified.')
            
    def get_life(self, C, k, algorithm = 'four-point',  Su = False, range = False, 
                 nr_load_classes=512, **kwargs):
        """Calculate fatigue life with parameters C, k, as defined in [3]
        
        :param C: [int,float]
            S-N curve intercept [MPa**k].
        :param k: [int,float]
            S-N curve inverse slope [/].
        :param algorithm: str
            Cycle counting method. Options are 'three-point' and 'four-point'.
            Defaults to 'four-point'.
        :param Su: [int,float]
            Ultimate tensile strength [MPa]. If specified, Goodman equivalent stress
            is used for fatigue life estimation. Defaults to False.
        :param range: bool
            If True, ranges instead of amplitudes are used for fatigue life estimation.
            Defaults to False.
        :param nr_load_classes: int
            The number of intervals to divide the min-max range of the dataseries
            into. Used with algorithm ='four-point'. 
            Defaults to 512.
        :returns:
            Estimated fatigue life in seconds.
        :rtype: float
        """    
        t = self.spectral_data.t
        cycles = self._get_cycles(algorithm = algorithm, Su=Su, **kwargs)
        ranges = cycles[0]
        means = cycles[1]
        
        if range == True: 
            pass
        elif range == False: 
            ranges *= 0.5
        else: 
            raise Exception('Unrecognized Input Error')

        if Su: ranges = ranges/(1. - means/Su)

        try :
            counts = cycles[2]  #three-point algorithm returns half cycles
            d = np.sum(counts * ranges**k / C)
        except Exception:
            d = np.sum(ranges**k / C)

        T = t / d
        
        return T

    def _get_cycles(self, algorithm = 'four-point', nr_load_classes=512, **kwargs):
        """
        :param algorithm: str
            Cycle counting method. Options are `three-point` and `four-point`.
            Defaults to `four-point`.
        :param nr_load_classes: int
            The number of intervals to divide the min-max range of the dataseries
            into. Used with algorithm =`four-point`. 
            Defaults to 512.
        :returns (ranges, means, counts): tuple of numpy.ndarray
            For algorithm = 'four-point', tuple with only two elemensts (ranges, means) is returned.
        """
        if algorithm == 'four-point':
            ranges, means = fatpack.find_rainflow_ranges(self.spectral_data.data, k=nr_load_classes, return_means=True)
            return ranges, means

        elif algorithm == 'three-point':
            cycles = np.array(list(rainflow.extract_cycles(self.spectral_data.data)))
            ranges = cycles[:,0]
            means = cycles[:,1]
            counts = cycles[:,2]
            return ranges, means, counts

        else: 
            raise ValueError('Set `algorithm` either to `three-point` or `four-point`.')