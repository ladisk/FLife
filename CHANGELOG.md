# Changelog

## 3.0 (2026-11-03)

### Breaking changes

- LiWI and COIN-LiWI and the multiaxial amplitude-spectrum input are removed (#70).
- `TovoBenasciutti.get_PDF` is normalised to 1 per cycle (#42, #44).
- Multiaxial results change for `max_normal_and_shear`, `max_normal`, `max_shear`, Carpinteri-Spagnoli (`cs`) and `multiaxial_rainflow` (#49, #50, #53, #54, #57).

### Added

- `EquivalentStress.gao_mwcm`, the spectral Modified Woehler Curve Method of Gao et al. (2021), zero mean stress (#69).
- `TovoBenasciutti.get_cycle_rate`, the number of cycles per second that goes with the normalised PDF (#44).
- `cs` takes the observation time `T` (#54).
- `multiaxial_rainflow` takes the S-N slope `k` (#57).
- Warnings from `Low.get_life` for negative large-cycle damage and for beta below 4 (#45).
- Warning from `JunPark` when D_2, D_3 or sigma_H is not positive (#47).
- Tests that each method's PDF integrates to 1 and matches its closed-form damage (#48).
- FLife logo, used in the README, the docs header and as the docs favicon (#76).

### Changed

- `max_normal`, `max_shear` and `max_normal_and_shear` search the critical plane on a grid and refine the best peaks, which is 8x or more faster per point (#67).
- `multiaxial_rainflow` uses the largest eigenvalue instead of a numerical search (4096 lines: 43 s to 5 ms) (#56).
- EVMS, thermoelastic, Lemaitre, Nieslony and `EVMS_out_of_phase` evaluate all points at once, about 20x to 60x faster for many points (#65, #66).
- The criteria EVMS, thermoelastic, Lemaitre, `max_normal`, `max_shear` and `max_normal_and_shear` return a real equivalent PSD (#62).
- `EquivalentStress` handles single and multiple points in one place, and `SpectralData` input checks are simpler. Results are unchanged (#72, #73).
- Showcase notebooks re-run with the current API (#75).
- README and docs show a new overview figure of the multiaxial criteria and spectral methods (`SpectralMethods.svg`), which replaces `FreqMethodsTree.png` (#77).
- Supported Python versions are 3.11 to 3.15.

### Fixed

Result changes:

- Tovo-Benasciutti: `get_life(..., integrate_pdf=True)` now agrees with the closed form. Before, the life was up to 13 % too long (#42, #44).
- `max_normal`, `max_shear` and `max_normal_and_shear` no longer depend on the load unit. Before, the found variance could drop to 67 % of the maximum for loads in Pa. The `max_normal_and_shear` test value moves from 1565.05 to 1746.62 (#49).
- `max_normal_and_shear` picks the plane with the largest equivalent stress variance. Uniaxial load now gives sigma, and rotating the load no longer changes the result (#50).
- Carpinteri-Spagnoli: the critical plane is searched over the whole sphere with a fixed tilt sign, so results can change (#53).
- Carpinteri-Spagnoli: the cycle count is nu_0 * T, not a rate. Test value 1647.03 to 1650.30 (+0.2 %) (#54).
- `multiaxial_rainflow` uses one fixed direction for the whole spectrum, as in Pitoiset and Preumont (2000), not one per frequency. Results were too high. Test value 1636.59 to 1621.01 (-1 %) (#57).
- `EVMS_out_of_phase` uses the largest component as phase reference. Before, the result was too high when the sigma_x PSD was zero (7 instead of 4 in one case) (#58).
- `multiaxial_rainflow` no longer depends on the stress unit (#55).

Other fixes:

- Nieslony criterion no longer returns NaN for rotated pure shear (#59).
- Carpinteri-Spagnoli uses the given frequency vector, which fixes a crash when `np.arange` gave one point more than the PSD (#52).
- `max_normal`, `max_shear` and `max_normal_and_shear` pass `search_method` on for multipoint input (#63).
- `EquivalentStress` passes `T`, `fs` and `rg` on, so seeded time histories are reproducible (#64).
- `EVMS_out_of_phase` raises a clear `ValueError` for a 3D PSD (#66).

### Deprecated

- `EquivalentStress.loop_over_points` raises a `DeprecationWarning` and will be removed in a future version. The criterion methods (for example `max_normal` or `EVMS`) handle multiple points themselves; call them instead (#71).

### Removed

- LiWI and COIN-LiWI multiaxial criteria (`EquivalentStress.liwi`, `EquivalentStress.coin_liwi`) and the multiaxial amplitude-spectrum input they needed. They work on deterministic complex harmonic amplitudes, not on random-process PSDs, so they do not belong in a spectral-methods package. Uniaxial amplitude-spectrum input to `SpectralData` is kept (#70).
- Unused `cplane.max_variance_old` (#74).
