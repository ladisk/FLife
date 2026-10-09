# Changelog

## 2.3.0

### Changed results

Many results change. Lives in seconds, comparison with 2.2.2.

Spectral band splitting (all multimodal methods):
- Band moments now include the last half-trapezoid. Moment m0 rises 0.04 to 1.2 %. Lives of the single-band methods fall 0.15 to 3 %, for example Dirlik on a narrowband PSD 5166 -> 5007 (-3.1 %).
- JiaoMoan, SakaiOkamura, FuCebon, ModifiedFuCebon, Low, Low2014, Lotsberg: band edges use the equal-area split of the trapezoid area. Change -0.1 to -6 %. FuCebon on the bundled multiaxial test point: -14.9 %.
- SingleMoment and BandsMethod now agree to 1e-10.

Method fixes:
- GaoMoan with 3 bands: true 2-D convolution of the large-cycle PDF. Lives fall 25 to 58 % for the test PSDs (up to 100 % for a low-frequency dominated PSD).
- TovoBenasciutti with `integrate_pdf=True`: alpha2 weight and peak rate fixed. Now equals the closed form. Change -3 to -13 %.
- Low: a warning is issued when the large-cycle damage is negative (narrowband PSDs, k >= 4). The value is unchanged.
- Low: a warning is issued when the band zero-crossing ratio beta = nu0_HF/nu0_LF (Low 2010, Eq. 15) is below 4. Low validated the method only for beta = 4 and 10. The value is unchanged.
- GaoMoan: the bandwidth parameters of the HF and MF bands in the large-cycle rates (Gao and Moan 2008, Eqs. 4 and 5) are set to zero, as recommended on p. 1947 ("It is therefore recommended to use zeros for these bandwidth parameters as done in this paper"). The article applies the procedure to bimodal processes too, so the 2-band variant uses delta_H = 0 as well. Lives rise 0 to 7 % for the standard test PSDs and 20 to 56 % for the bimodal PSD (10-20 + 150-170 Hz); the stored Welch test case goes 1019.6 -> 2483.6 s.

Amplitude spectrum input:
- PSD = |A|^2/df for the uniaxial and the multiaxial path. LiWI and COIN-LiWI equivalent PSD is divided by df. On the bundled test data (df = 3 Hz) the PSD falls by 3 and the life by 3^(k/2), about 55 times for k = 7.3.
- Amplitude input now works with all multiaxial criteria, not only LiWI.
- A is the RMS amplitude per frequency line. For one-sided peak (FFT) amplitudes, as in Schmidt and Pandiya (2021) and Schmidt and Kraft (2023), pass A/sqrt(2).

Multiaxial criteria (bundled test data, ratio of equivalent m0 new/old):
- max_normal_and_shear: critical plane is the plane of maximum equivalent-stress variance (Bedkowski and Macha). Median +3.3 %, up to +109 %. Reference test point +65 %.
- cs: N1 = nu0*T with nu0 in Hz (new argument `T`). T is taken from `cs(T=...)`, else from the T of EquivalentStress or SpectralData; if neither is set, a ValueError is raised. Delta limits as in the authors' code, new plane search. Median +0.3 %, range 0.69 to 2.0.
- max_normal, max_shear: same planes at normal load levels. The old search lost 11 to 29 % of the variance at PSD levels above 1e8 or below 1e-8.
- multiaxial_rainflow: one fixed direction c for the whole spectrum (Pitoiset and Preumont 2000, Eqs. 9-14, Sec. 3 and 4.1), chosen by maximum single-moment damage. New argument `k` (S-N slope); without `k` the maximum-variance direction is used and a UserWarning is issued. The old result was the frequency-wise maximum, which is not the PSD of any c^T s and always overestimates. Reference test point (k = 7.3): 1636.59 -> 1620.70 (-1.0 %). The chosen c is stored in `multiaxial_rainflow_direction`.
- EVMS, thermoelastic, Lemaitre: real output (was complex for a single point).
- EVMS_out_of_phase: a 6x6 PSD raises ValueError (it gave wrong results before).
- psd_trim_length: now the number of points kept, as documented. Before it was scaled by df and also cut the columns.

### Fixed

- TovoBenasciutti `get_PDF`: weights are now b*alpha2 for the level-crossing term and (1-b) for the range-counting term (Benasciutti and Tovo 2005, Eq. 28). It is a per-peak density that integrates to 1 - b(1-alpha2), the rest is zero-amplitude mass. Lives from `get_life` are unchanged; damage from `nu_p/C * int s^k get_PDF ds` equals the closed form of Eq. 27.
- JiaoMoan and GaoMoan with 1 or 2 bands no longer raise TypeError.
- `rg` is passed to time-history generation, so a seed gives the same signal.
- Input checks: f increasing and non-negative, PSD non-negative, m0 > 0, uniform f for multiaxial input, correct shape and fs checks.
- LiWI and COIN-LiWI single-point input no longer raises AttributeError and no longer overwrites the PSD per frequency.
- COIN-LiWI phase now follows Schmidt and Kraft (2023), Eq. 36. The old code picked the wrong branch at about 40 % of frequencies. The equivalent PSD is unchanged.
- LiWI phase indicator: the cross term of the resultant amplitude now follows Schmidt and Pandiya (2021), Eqs. 15, 17 and 24 (2 A1 A2 cos(theta) e^{j(phi_1+phi_2)}). The phase now equals the article's exact-equation phase. The old code picked the other branch for about 4 % of random plane-stress states. The amplitude and the equivalent PSD are unchanged.
- LiWI amplitude: no change. It is the Bonte amplitude of Schmidt and Pandiya (2021), Eqs. 10-11, which equals COIN-LiWI with k_a^2 = 3 and k_phi = 0.5. Schmidt and Kraft (2023), Eq. 38, uses a revised LiWI with k_phi = 1, available as `coin_liwi(k_a=sqrt(3), k_phi=1)`.
- Rainflow: `T` and `fs` kwargs check, `nr_load_classes` honoured. Goodman raises for mean >= Su; a mean <= -Su is computed normally (compressive mean raises life).
- JunPark warning condition. PSDgen save format and rounding.

### Performance

- Low.get_life: 10-15 s -> 0.01 s per call (Gauss-Legendre grids, same values to 1e-10).
- Critical-plane criteria on all points at once: cs 7x, max_normal_and_shear 95x, max_shear 260x, max_normal 716x per point.
- Other criteria: EVMS, thermoelastic, Lemaitre 23 to 24x; Nieslony, EVMS_out_of_phase, liwi 74 to 103x; coin_liwi 200x; multiaxial_rainflow about 6900x.
- SpectralData construction 0.105 s -> 0.074 s for 200k points. random_gaussian with random_amplitude 4.7 s -> 0.05 s for 2e6 points.

### Packaging

- pyExSi, lvm_read, pytest and pylint are no longer runtime dependencies. pyExSi and lvm_read are in the `dev` extra.
- New `gui` extra: pyvista, pyvistaqt, PyQt5. New `dev` extra: pytest, pytest-cov, pylint and build tools.
- `visualize`, `set_mesh`, `pick_point` and `PSDgen` are imported lazily. `import FLife` works without GUI packages. GUI input without tkinter raises an ImportError with install instructions.
- CI installs `.[dev,gui]` and gates the release on tests.

### API

- `Rainflow.get_life` with `Su=0` (or any `Su <= 0`) now raises ValueError.
- `EquivalentStress.loop_over_points` is deprecated and issues a DeprecationWarning. The criteria process all points at once.
- `PSDgen` is no longer in `FLife.__all__`. `FLife.PSDgen` still works.

### Known open issues

- C-S Eq. 22 (squaring of the s_af/tau_af term) is on hold. The code keeps the linear form, as in the authors' code.
- C-S result still depends on the reference frame for non-proportional loading (general 3D state 0.978 after rotation, pure shear 0.74 to 1.48).
- LiWI phase sign depends on the in-plane reference frame for non-proportional states (sign flips for 19 to 60 % of random states after rotation). This is also true for the exact equations of Schmidt and Pandiya (2021), so it is not a code error. The amplitude and the equivalent PSD are invariant. COIN-LiWI (phase from the first invariant, Schmidt and Kraft 2023, Eq. 36) has an invariant phase.
