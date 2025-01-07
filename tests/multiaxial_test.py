import numpy as np
import sys, os
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
        'max_normal': 2759.824771658108,
        'max_shear': 2759.824771144651,
        'max_normal_and_shear': 2759.824771144652, #s_af = 1, tau_af = 1
        'cs': 931.3556831975611, #s_af = 1, tau_af = 1
        'multiaxial_rainflow': 1636.5919423074781,
        'thermoelastic': 1027.6423513625518,
        'liwi': 2083.2568582803156,
        'coin_liwi': 2044.550993026664, #k_a=1.70, k_phi=0.90
        'EVMS_out_of_phase': 2083.2568582803156,
        'Nieslony': 1592.0621836751761, #s_af = 1, tau_af = 1
        'Lemaitre': 1942.5082573579468 # poisson_ratio = 0.5
    }

    #test_PSD
    test_PSD = np.load('data/test_multiaxial_PSD_3D.npy')
    test_PSD_biaxial = np.load('data/test_multiaxial_PSD_2D.npy')

    #Amplitude spectrum inputs
    test_amplitude_spectrum_2D = np.load('data/test_multiaxial_amplitude_spectrum_2D.npy')
    test_amplitude_spectrum_3D = np.load('data/test_multiaxial_amplitude_spectrum_3D.npy')

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

    for criterion, value in results.items():

        np.testing.assert_almost_equal(value, results_ref[criterion], decimal=5, err_msg=f'Criterion: {criterion}')

if __name__ == "__main__":
    test_data()
    #test_version()

if __name__ == '__mains__':
    np.testing.run_module_suite()