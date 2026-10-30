"""Conversion between the amplitude spectrum and the PSD: PSD = |A|^2/df (RMS amplitudes)."""
import numpy as np
import pytest
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
from FLife.tools import amplitude_spectrum_to_psd, psd_to_amplitude_spectrum

F = np.arange(0., 120., 2.)
DF = F[1] - F[0]


def amplitudes(shape, seed=0):
    """Random complex amplitude spectrum, frequency on the axis given by F.size in shape."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def same_up_to_phase(a, b):
    """Amplitudes a and b at each frequency differ only by a common phase."""
    phase = np.sum(a * np.conj(b), axis=-1, keepdims=True)
    np.testing.assert_allclose(a, b * phase / np.abs(phase), rtol=1e-8)


# --- uniaxial ---------------------------------------------------------------------------

def test_uniaxial_psd_value():
    psd = amplitude_spectrum_to_psd(3 * np.ones(F.size), F)
    np.testing.assert_allclose(psd, 9 / DF)


def test_uniaxial_complex_amplitude_gives_real_psd():
    a = amplitudes(F.size)
    psd = amplitude_spectrum_to_psd(a, F)
    assert np.isrealobj(psd)
    np.testing.assert_allclose(psd, np.abs(a)**2 / DF)


def test_uniaxial_variance_is_sum_of_squared_rms_amplitudes():
    a = amplitudes(F.size)
    psd = amplitude_spectrum_to_psd(a, F)
    np.testing.assert_allclose(np.sum(psd) * DF, np.sum(np.abs(a)**2))


def test_uniaxial_round_trip():
    a = np.abs(amplitudes(F.size))
    np.testing.assert_allclose(psd_to_amplitude_spectrum(amplitude_spectrum_to_psd(a, F), F), a)


def test_uniaxial_matches_spectral_data():
    a = np.abs(amplitudes(F.size))
    sd = FLife.SpectralData(input={'amplitude_spectrum': a, 'f': F})
    np.testing.assert_allclose(sd.psd[:, 1], amplitude_spectrum_to_psd(a, F))


def test_peak_amplitude_has_half_the_power():
    a = np.abs(amplitudes(F.size))
    np.testing.assert_allclose(amplitude_spectrum_to_psd(a, F, amplitude='peak'),
                               amplitude_spectrum_to_psd(a, F) / 2)
    np.testing.assert_allclose(psd_to_amplitude_spectrum(amplitude_spectrum_to_psd(a, F, amplitude='peak'), F, amplitude='peak'), a)


# --- multiaxial -------------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(F.size, 3), (F.size, 6), (4, F.size, 3), (4, F.size, 6)])
def test_multiaxial_psd_matrix(shape):
    a = amplitudes(shape)
    s = amplitude_spectrum_to_psd(a, F)
    assert s.shape == shape + (shape[-1],)
    np.testing.assert_allclose(s, np.conj(np.swapaxes(s, -1, -2)))
    np.testing.assert_allclose(np.real(np.diagonal(s, axis1=-2, axis2=-1)), np.abs(a)**2 / DF)
    np.testing.assert_allclose(s[..., 0, 1], a[..., 0] * np.conj(a[..., 1]) / DF)


@pytest.mark.parametrize('amplitude', ['rms', 'peak'])
@pytest.mark.parametrize('shape', [(F.size, 3), (F.size, 6), (4, F.size, 3), (4, F.size, 6)])
def test_multiaxial_round_trip(shape, amplitude):
    a = amplitudes(shape)
    s = amplitude_spectrum_to_psd(a, F, amplitude=amplitude)
    back = psd_to_amplitude_spectrum(s, F, amplitude=amplitude)
    assert back.shape == shape
    same_up_to_phase(back, a)
    np.testing.assert_allclose(amplitude_spectrum_to_psd(back, F, amplitude=amplitude), s, atol=1e-9)


def test_multiaxial_largest_component_is_real_and_positive():
    back = psd_to_amplitude_spectrum(amplitude_spectrum_to_psd(amplitudes((F.size, 6)), F), F)
    largest = np.take_along_axis(back, np.argmax(np.abs(back), axis=-1)[..., None], axis=-1)
    assert np.all(np.real(largest) > 0)
    np.testing.assert_allclose(np.imag(largest), 0, atol=1e-12)


def test_multiaxial_real_psd_gives_real_amplitudes():
    s = amplitude_spectrum_to_psd(np.real(amplitudes((F.size, 3))), F)
    assert np.isrealobj(s)
    assert np.isrealobj(psd_to_amplitude_spectrum(s, F))


def test_multiaxial_zero_lines_stay_zero():
    a = amplitudes((F.size, 3))
    a[10] = 0
    back = psd_to_amplitude_spectrum(amplitude_spectrum_to_psd(a, F), F)
    assert np.all(np.isfinite(back))
    np.testing.assert_allclose(back[10], 0, atol=1e-12)


@pytest.mark.parametrize('n_comp, method, args', [(3, 'liwi', {}), (6, 'coin_liwi', dict(k_a=1.7, k_phi=0.9))])
def test_criteria_do_not_depend_on_the_common_phase(n_comp, method, args):
    # the recovered amplitudes lose only the common phase, which LiWI and COIN-LiWI do not use
    a = amplitudes((3, F.size, n_comp))
    back = psd_to_amplitude_spectrum(amplitude_spectrum_to_psd(a, F), F)
    original = FLife.EquivalentStress(input={'amplitude_spectrum': a, 'f': F})
    recovered = FLife.EquivalentStress(input={'amplitude_spectrum': back, 'f': F})
    getattr(original, method)(**args)
    getattr(recovered, method)(**args)
    np.testing.assert_allclose(recovered.eq_psd_multipoint[0], original.eq_psd_multipoint[0], rtol=1e-8)


@pytest.mark.parametrize('n_comp', [3, 6])
def test_bundled_psd_is_coherent(n_comp):
    # the bundled PSD matrices are rank 1, so they convert to an amplitude spectrum
    psd = np.load(os.path.join(my_path, '..', 'data', f'test_multiaxial_PSD_{2 if n_comp == 3 else 3}D.npy'))[331:335]
    f = np.arange(0, 240, 3)
    a = psd_to_amplitude_spectrum(psd, f)
    assert a.shape == (4, f.size, n_comp)
    np.testing.assert_allclose(amplitude_spectrum_to_psd(a, f), psd, atol=1e-6 * np.abs(psd).max())


# --- input errors -----------------------------------------------------------------------

def test_incoherent_psd_matrix_raises():
    s = amplitude_spectrum_to_psd(amplitudes((F.size, 3), seed=1), F)
    s = s + amplitude_spectrum_to_psd(amplitudes((F.size, 3), seed=2), F)
    with pytest.raises(ValueError, match='coherent'):
        psd_to_amplitude_spectrum(s, F)


def test_non_hermitian_psd_matrix_raises():
    s = amplitude_spectrum_to_psd(amplitudes((F.size, 3)), F)
    s[5, 0, 1] += 1.
    with pytest.raises(ValueError, match='Hermitian'):
        psd_to_amplitude_spectrum(s, F)


def test_negative_psd_raises():
    psd = np.ones(F.size)
    psd[3] = -1.
    with pytest.raises(ValueError, match='non-negative'):
        psd_to_amplitude_spectrum(psd, F)


@pytest.mark.parametrize('f', [np.arange(5.), np.array([0., 1., 3., 4.] + list(np.arange(5., 61.))), F[::-1]])
def test_bad_frequency_vector_raises(f):
    with pytest.raises(ValueError):
        amplitude_spectrum_to_psd(np.ones(F.size), f)
    with pytest.raises(ValueError):
        psd_to_amplitude_spectrum(np.ones(F.size), f)


@pytest.mark.parametrize('shape', [(F.size, 3, 2), (2, 2, F.size, 3, 3)])
def test_bad_shape_raises(shape):
    with pytest.raises(ValueError):
        psd_to_amplitude_spectrum(np.ones(shape), F)


def test_unknown_amplitude_type_raises():
    with pytest.raises(ValueError, match='rms'):
        amplitude_spectrum_to_psd(np.ones(F.size), F, amplitude='rmsd')
