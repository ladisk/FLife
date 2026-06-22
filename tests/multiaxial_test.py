import numpy as np
import sys, os
import pytest
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

def test_version():
    """ check FLife exposes a version attribute """
    assert hasattr(FLife, '__version__')
    assert isinstance(FLife.__version__, str)

#reference results
def test_data():
    results_ref = {
        'EVMS': 2083.2568582803156,
        'max_normal': 1650.300624416056,
        'max_shear': 2430.1468521282327,
        'max_normal_and_shear': 1565.051025189015, #s_af = 1, tau_af = 1
        'cs': 1647.0269200556531, #s_af = 1, tau_af = 1 (optimiser-based)
        'multiaxial_rainflow': 1636.5919423074781,
        'thermoelastic': 1027.6423513625518,
        'liwi': 2083.2568582803156,
        'coin_liwi': 2044.550993026664, #k_a=1.70, k_phi=0.90
        'EVMS_out_of_phase': 2083.2568582803156,
        'Nieslony': 1592.0621836751761, #s_af = 1, tau_af = 1
        'Lemaitre': 1942.5082573579468 # poisson_ratio = 0.3
    }

    #test_PSD
    data_dir = os.path.join(my_path, '..', 'data')
    test_PSD = np.load(os.path.join(data_dir, 'test_multiaxial_PSD_3D.npy'))
    test_PSD_biaxial = np.load(os.path.join(data_dir, 'test_multiaxial_PSD_2D.npy'))

    #Amplitude spectrum inputs
    test_amplitude_spectrum_2D = np.load(os.path.join(data_dir, 'test_multiaxial_amplitude_spectrum_2D.npy'))
    test_amplitude_spectrum_3D = np.load(os.path.join(data_dir, 'test_multiaxial_amplitude_spectrum_3D.npy'))

    freq=np.arange(0,240,3)
    freq[0] = 1e-3

    input_dict_psd_3d = {'PSD': test_PSD[331:335], 'f': freq}
    input_dict_psd_2d = {'PSD': test_PSD_biaxial[331:335], 'f': freq}

    input_dict_ampl_3d = {'amplitude_spectrum': test_amplitude_spectrum_3D[331:335], 'f': freq}
    input_dict_ampl_2d = {'amplitude_spectrum': test_amplitude_spectrum_2D[331:335], 'f': freq}

    C = 1.8e+22
    k = 7.3

    # Equivalent Stress
    EVMS = FLife.EquivalentStress(input=input_dict_psd_3d,T=1,fs=5000)
    max_normal = FLife.EquivalentStress(input=input_dict_psd_3d,T=1,fs=5000)
    max_shear = FLife.EquivalentStress(input=input_dict_psd_3d,T=1,fs=5000)
    max_normal_and_shear = FLife.EquivalentStress(input=input_dict_psd_3d,T=1,fs=5000)
    cs = FLife.EquivalentStress(input=input_dict_psd_3d,T=1,fs=5000)
    multiaxial_rainflow = FLife.EquivalentStress(input=input_dict_psd_2d,T=1,fs=5000)
    thermoelastic = FLife.EquivalentStress(input=input_dict_psd_3d,T=1,fs=5000)
    liwi = FLife.EquivalentStress(input=input_dict_ampl_2d,T=1,fs=5000)
    coin_liwi = FLife.EquivalentStress(input=input_dict_ampl_3d,T=1,fs=5000)
    EVMS_out_of_phase = FLife.EquivalentStress(input=input_dict_psd_2d,T=1,fs=5000)
    Nieslony = FLife.EquivalentStress(input=input_dict_psd_3d,T=1,fs=5000)
    Lemaitre = FLife.EquivalentStress(input=input_dict_psd_3d,T=1,fs=5000)

    # Multiaxial criteria
    EVMS.EVMS()
    max_normal.max_normal()
    max_shear.max_shear()
    max_normal_and_shear.max_normal_and_shear(s_af=1, tau_af=1)
    cs.cs(s_af=1, tau_af=1)
    multiaxial_rainflow.multiaxial_rainflow()
    thermoelastic.thermoelastic()
    liwi.liwi()
    coin_liwi.coin_liwi(k_a=1.70, k_phi=0.90)
    EVMS_out_of_phase.EVMS_out_of_phase()
    Nieslony.Nieslony(s_af=1, tau_af=1)
    Lemaitre.Lemaitre(poisson_ratio=0.3)

    results = {
        'EVMS': EVMS.eq_psd_multipoint[0][0,24],
        'max_normal': max_normal.eq_psd_multipoint[0][0,24],
        'max_shear': max_shear.eq_psd_multipoint[0][0,24],
        'max_normal_and_shear': max_normal_and_shear.eq_psd_multipoint[0][0,24],
        'cs': cs.eq_psd_multipoint[0][0,24],
        'multiaxial_rainflow': multiaxial_rainflow.eq_psd_multipoint[0][0,24],
        'thermoelastic': thermoelastic.eq_psd_multipoint[0][0,24],
        'liwi': liwi.eq_psd_multipoint[0][0,24],
        'coin_liwi': coin_liwi.eq_psd_multipoint[0][0,24],
        'EVMS_out_of_phase': EVMS_out_of_phase.eq_psd_multipoint[0][0,24],
        'Nieslony': Nieslony.eq_psd_multipoint[0][0,24],
        'Lemaitre': Lemaitre.eq_psd_multipoint[0][0,24]
    }

    # criteria whose critical plane is found by numerical optimisation are only
    # reproducible to a relative tolerance across platforms / scipy versions
    optimiser_based = {'max_normal', 'max_shear', 'max_normal_and_shear', 'cs'}

    for criterion, value in results.items():

        if criterion in optimiser_based:
            np.testing.assert_allclose(value, results_ref[criterion], rtol=1e-4, err_msg=f'Criterion: {criterion}')
        else:
            np.testing.assert_almost_equal(value, results_ref[criterion], decimal=5, err_msg=f'Criterion: {criterion}')


def _plane_stress_psd_to_voigt_3d(psd_2d):
    """Embed plane-stress PSD [sx, sy, txy] into 3D Voigt [sx, sy, sz, txy, txz, tyz]."""
    psd_3d = np.zeros(psd_2d.shape[:-2] + (6, 6), dtype=psd_2d.dtype)
    plane_stress_indices = [0, 1, 3]

    for row_2d, row_3d in enumerate(plane_stress_indices):
        for col_2d, col_3d in enumerate(plane_stress_indices):
            psd_3d[..., row_3d, col_3d] = psd_2d[..., row_2d, col_2d]

    return psd_3d


def _run_psd_criterion(input_dict, criterion_name, **kwargs):
    eq_stress = FLife.EquivalentStress(input=input_dict, T=1, fs=5000)
    getattr(eq_stress, criterion_name)(**kwargs)
    return eq_stress.eq_psd_multipoint[0]


def test_plane_stress_reference_psd_matches_zero_padded_3d_reference():
    data_dir = os.path.join(my_path, '..', 'data')
    test_psd_2d = np.load(os.path.join(data_dir, 'test_multiaxial_PSD_2D.npy'))
    test_psd_3d = np.load(os.path.join(data_dir, 'test_multiaxial_PSD_3D.npy'))

    np.testing.assert_allclose(_plane_stress_psd_to_voigt_3d(test_psd_2d), test_psd_3d)


@pytest.mark.parametrize(
    ('criterion_name', 'criterion_kwargs'),
    [
        ('max_normal', {}),
        ('max_shear', {}),
        ('max_normal_and_shear', {'s_af': 1, 'tau_af': 1}),
        ('cs', {'s_af': 1, 'tau_af': 1}),
        ('Nieslony', {'s_af': 1, 'tau_af': 1}),
    ],
)
def test_plane_stress_psd_criteria_match_zero_padded_3d(criterion_name, criterion_kwargs):
    """Plane-stress PSD input should behave like the equivalent 3D Voigt PSD with zero out-of-plane terms."""
    data_dir = os.path.join(my_path, '..', 'data')
    test_psd_2d = np.load(os.path.join(data_dir, 'test_multiaxial_PSD_2D.npy'))

    freq=np.arange(0,240,3)
    freq[0] = 1e-3

    input_dict_psd_2d = {'PSD': test_psd_2d[331:335], 'f': freq}
    input_dict_psd_3d = {'PSD': _plane_stress_psd_to_voigt_3d(test_psd_2d[331:335]), 'f': freq}

    psd_2d_eq = _run_psd_criterion(input_dict_psd_2d, criterion_name, **criterion_kwargs)
    psd_3d_eq = _run_psd_criterion(input_dict_psd_3d, criterion_name, **criterion_kwargs)

    np.testing.assert_allclose(psd_2d_eq, psd_3d_eq, rtol=1e-6, atol=1e-8)

if __name__ == "__main__":
    test_data()
    #test_version()

if __name__ == '__mains__':
    np.testing.run_module_suite()
