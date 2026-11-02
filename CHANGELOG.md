# Changelog

## Unreleased

### Changed results

- Amplitude spectrum input is now converted to a PSD as |A|^2/df, where A is the RMS amplitude
  per frequency line. Uniaxial input already used this. The multiaxial LiWI and COIN-LiWI
  criteria used |A|^2 without /df in 2.3. Complex uniaxial amplitudes now give a real PSD.
- LiWI and COIN-LiWI lives from amplitude spectrum input change by a factor df^(k/2) versus 2.3
  (k is the S-N curve inverse slope). PSD input is not affected.
- The bundled multiaxial amplitude spectra in `data/` are regenerated as A*sqrt(df), so that they
  describe the same stress state as the bundled PSD files (PSD = A A^T/df). Results from these
  files with LiWI and COIN-LiWI are the same as in 2.3.

### Added

- `FLife.tools.amplitude_spectrum_to_psd` and `FLife.tools.psd_to_amplitude_spectrum` convert
  between an amplitude spectrum and a PSD. Use `amplitude='peak'` for one-sided peak amplitudes.
