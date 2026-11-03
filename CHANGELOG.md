# Changelog

## Unreleased

### Removed

- LiWI and COIN-LiWI multiaxial criteria (`EquivalentStress.liwi`, `EquivalentStress.coin_liwi`) and the multiaxial amplitude-spectrum input they needed. They work on deterministic complex harmonic amplitudes, not on random-process PSDs, so they do not belong in a spectral-methods package. Uniaxial amplitude-spectrum input to `SpectralData` is kept.
