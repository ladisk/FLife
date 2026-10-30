import numpy as np
from scipy import stats
from scipy.special import ndtr

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
            * (ndtr(np.sqrt(m0L_norm/m0H_norm)*x/np.sqrt(m0)) + ndtr(np.sqrt(m0H_norm/m0L_norm)*x/np.sqrt(m0)) - 1))
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
    
    if rg == None:
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


def _spectrum_scale(f, amplitude, n_lines):
    """Returns the factor c in PSD = c*|A|^2, for amplitudes of the given type."""
    f = np.asarray(f)
    if f.ndim != 1 or f.size < 2:
        raise ValueError('Frequency vector `f` should be 1D with at least 2 points.')
    if f.size != n_lines:
        raise ValueError(f'Length of `f` ({f.size}) does not match the frequency axis of the input ({n_lines}).')
    df = np.diff(f)
    if np.any(df <= 0) or not np.allclose(df, np.median(df), rtol=1e-3):
        raise ValueError('Frequency vector `f` should be uniform and strictly increasing.')
    if amplitude == 'rms':
        return 1 / (f[1] - f[0])
    if amplitude == 'peak':
        return 1 / (2 * (f[1] - f[0]))
    raise ValueError("`amplitude` should be 'rms' or 'peak'.")


def amplitude_spectrum_to_psd(amplitude_spectrum, f, amplitude='rms'):
    """Converts an amplitude spectrum to a PSD: PSD = |A|^2/df for RMS amplitudes
    and PSD = |A|^2/(2*df) for one-sided peak amplitudes.

    A multiaxial amplitude spectrum gives the PSD matrix S_ij = A_i*conj(A_j)/df.

    :param amplitude_spectrum: array_like
        Real or complex amplitude spectrum with the shape (f,) - uniaxial, (f,n) - multiaxial
        at a single point or (N,f,n) - multiaxial at N points.
    :param f: array_like
        Uniform frequency vector [Hz].
    :param amplitude: str
        Type of the amplitudes: 'rms' (default) or 'peak'.
    :return psd: numpy.ndarray
        Real PSD with the shape (f,) for uniaxial input, otherwise Hermitian PSD matrix
        with the shape (f,n,n) or (N,f,n,n).
    """
    a = np.asarray(amplitude_spectrum)
    if a.ndim not in (1, 2, 3):
        raise ValueError('Amplitude spectrum should have the shape (f,), (f,n) or (N,f,n).')
    c = _spectrum_scale(f, amplitude, a.shape[1] if a.ndim == 3 else a.shape[0])
    if a.ndim == 1:
        return c * np.abs(a)**2
    return c * a[..., :, None] * np.conj(a[..., None, :])


def psd_to_amplitude_spectrum(psd, f, amplitude='rms', rank_tol=1e-6):
    """Converts a PSD to an amplitude spectrum: A = sqrt(PSD*df) for RMS amplitudes
    and A = sqrt(2*PSD*df) for one-sided peak amplitudes.

    A PSD holds no absolute phase, so a uniaxial result is real and non-negative, and a multiaxial
    result is defined up to a common phase at each frequency: the largest component is real and positive.
    A multiaxial amplitude spectrum exists only for a fully coherent PSD matrix (rank 1 at each frequency),
    otherwise ValueError is raised.

    :param psd: array_like
        PSD with the shape (f,) - uniaxial, or Hermitian PSD matrix with the shape (f,n,n)
        - single point or (N,f,n,n) - N points.
    :param f: array_like
        Uniform frequency vector [Hz].
    :param amplitude: str
        Type of the amplitudes: 'rms' (default) or 'peak'.
    :param rank_tol: float
        Largest allowed ratio between the other eigenvalues and the largest eigenvalue of the PSD matrix.
    :return amplitude_spectrum: numpy.ndarray
        Amplitude spectrum with the shape (f,), (f,n) or (N,f,n).
    """
    s = np.asarray(psd)
    if s.ndim not in (1, 3, 4) or (s.ndim > 1 and s.shape[-1] != s.shape[-2]):
        raise ValueError('PSD should have the shape (f,), (f,n,n) or (N,f,n,n).')
    c = _spectrum_scale(f, amplitude, s.shape[1] if s.ndim == 4 else s.shape[0])

    if s.ndim == 1:
        if np.min(s) < -1e-8 * np.max(np.abs(s)):
            raise ValueError('PSD values should be non-negative.')
        return np.sqrt(np.clip(s, 0, None) / c)

    if not np.allclose(s, np.conj(np.swapaxes(s, -1, -2))):
        raise ValueError('PSD matrix should be Hermitian.')
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    largest = eigenvalues[..., -1]
    if np.any(np.abs(eigenvalues[..., :-1]).max(axis=-1) > rank_tol * largest):
        raise ValueError('PSD matrix is not fully coherent (rank 1), it has no equivalent amplitude spectrum.')
    v = eigenvectors[..., -1]
    reference = np.take_along_axis(v, np.argmax(np.abs(v), axis=-1)[..., None], axis=-1)
    a = v * np.conj(reference) / np.abs(reference) * np.sqrt(np.clip(largest, 0, None) / c)[..., None]
    return a if np.iscomplexobj(s) else np.real(a)


def __getattr__(name):
    if name == "PSDgen":
        from .gui import PSDgen
        return PSDgen
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
