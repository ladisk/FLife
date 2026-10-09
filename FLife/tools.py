import numpy as np
from scipy import stats

def relative_error(value, value_ref):
    """Return the relative error 

    :param value: [int,float]
        Value to compare
    :param value: [int,float]
        Reference value
    :return err: float
        Relative error.
    """ 
    return (value-value_ref)/value_ref

def basquin_to_sn(Sf, b, range = False):
    """Converts Basquin equation parameters Sf and b to fatigue life parameters C and k,
    as defined in [2]. Basic form of Basquin equation is used here: Sa = Sf* (2*N)**k 

    :param Sf: [int,float]
        Fatigue strength coefficient [MPa**k].
    :param b : [int,float]
        Fatigue strength exponent [/]. Represents S-N curve slope.
    :param range : bool
        False/True sets returned value C with regards to amplitude / range count, respectively.
    :return C, k: float
        C - S-N curve intercept [MPa**k], k - S-N curve inverse slope [/].
    """ 
    if not range:
        k = -1/b 
        C = 0.5*Sf**k
        return C,k

    else: 
        k = -1/b 
        C = 0.5*(2*Sf)**k
        return C,k

def pdf_rayleigh_sum(m0L, m0H):
    """Returns PDF (Probability Density Function) of sum of 2 Rayleigh distributed RV (Random Variables).

    :param m0L, moH: [int,float]
        Variance of first and second Rayleigh distributed RV. 
    :return pdf: function
    """
    m0 = m0L + m0H
    m0L_norm = m0L/m0
    m0H_norm = m0H/m0

    def pdf(x):
        pdf = 1 / np.sqrt(m0) * (m0L_norm * x/np.sqrt(m0) * np.exp(- (x/np.sqrt(m0))**2 /(2*m0L_norm)) \
            + m0H_norm * x/np.sqrt(m0) * np.exp(- (x/np.sqrt(m0))**2 /(2*m0H_norm)) \
            + np.sqrt(2*np.pi*m0L_norm*m0H_norm) * ((x/np.sqrt(m0))**2 -1) * np.exp(-(x/np.sqrt(m0))**2/2) \
            * (stats.norm.cdf(np.sqrt(m0L_norm/m0H_norm)*x/np.sqrt(m0)) + stats.norm.cdf(np.sqrt(m0H_norm/m0L_norm)*x/np.sqrt(m0)) - 1))
        return pdf
    return pdf


def random_gaussian(freq, PSD, T, fs, rg=None, random_amplitude=False, **kwargs):
    """
    Stationary Gaussian realization of random process, characterized by PSD.
    
    Random process is obtained with IFFT of amplitude spectra with random phase [1]. Area under PSD curve represents variance of random process.
    
    :param freq: array
        Frequency vector
    :param PSD: array
        One-sided power spectral density [unit^2].
    :param T: int,float
        Length of returned random process [s].
    :param fs: int,float
        Sampling frequency [Hz].
    :param rg: numpy.random._generator.Generator
        Initialized Generator object
    :param random_amplitude: Boolean
        If true, Rayleigh distributed amplitude is used in addition 
        to uniformly distributed phase. Defaults to False
    :return: t, signal
        Time and stationary Gaussian realization of random process
    
    Notes
    -----
    PSD data points are interpolated using numpy.interp() function. Option for 
    interpolation function will be added in the future.
    
    References
    ----------
    [1] D. E. Newland. An Introduction to Random Vibrations, Spectral & Wavelet Analysis.
    Dover Publications, 2005
    """ 
    # time and frequency data
    if np.__version__>='2.0.0':
        trapezoid = np.trapezoid
    else:
        trapezoid = np.trapz
    var = trapezoid(PSD,freq)
    N = int(T * fs)
    M = N//2 + 1
    t = np.arange(0,N) / fs # time vector

    M = N//2 + 1
    freq_new = np.arange(0, M, 1) / N  * fs # frequency vector
    PSD_new = np.interp(freq_new, freq, PSD, left=0, right=0)
    
    if rg is None:
        rg = np.random.default_rng()
    
    if isinstance(rg, np.random._generator.Generator):
        
        phase = rg.uniform(0, 1, len(PSD_new))
        
        ampl_spectra = np.sqrt(PSD_new * N * fs / 2)  # amplitude spectra modulus
        if random_amplitude == True:
            ampl_spectra = rg.rayleigh(scale=ampl_spectra) # Rayleigh distributed amplitudes

        ampl_spectra_random = ampl_spectra * np.exp(
            1j * phase * 2 * np.pi)  # amplitude spectra,  random phase

    else:
        raise ValueError(
            '`rg` must be initialized Generator object (numpy.random._generator.Generator)!'
        )

    signal = np.fft.irfft(ampl_spectra_random, n=N)  # time signal
    signal = signal/np.std(signal) * var**.5
    return t, signal


def __getattr__(name):
    if name == "PSDgen":
        from .gui import PSDgen
        return PSDgen
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
