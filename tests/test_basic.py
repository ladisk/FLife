import numpy as np
from scipy.integrate import quad
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import lvm_read
import FLife

def test_version():
    """ check FLife exposes a version attribute """
    assert hasattr(FLife, '__version__')
    assert isinstance(FLife.__version__, str)

def test_data():
    results_ref = {
        'Rainflow': 906.217537,
        'Rainflow-Goodman': 827.866874,
        'Dirlik': 1067.423788,
        'Tovo Benasciutti 1': 735.084318,
        'Tovo Benasciutti 2': 1114.625812,
        'Zhao Baker 1': 985.886435,
        'Zhao Baker 2': 1048.549852,
        'Narrowband': 711.258072,
        'Alpha 0.75': 1086.593252,
        'Wirsching Light': 1038.1813918800456,
        'Rice': 687.739914,
        'Gao Moan': 837.392263,
        'Petrucci Zuccarello': 4.322102
    }

    data = lvm_read.read('./data/m1.lvm')[0]

    t = data['data'][:,0]
    x = data['data'][:,1]

    rms = 100  
    C = 1.8e+22
    k = 7.3
    Su = 446
    x = rms * x / np.std(x) 

    # Spectral data
    sd = FLife.SpectralData(input=x, dt=t[1], nperseg=int(0.1/t[1]))

    # Rainflow reference fatigue life
    rf = FLife.Rainflow(sd)

    # Spectral methods
    dirlik = FLife.Dirlik(sd)
    tb = FLife.TovoBenasciutti(sd)
    zb = FLife.ZhaoBaker(sd)
    nb = FLife.Narrowband(sd)
    a075 = FLife.Alpha075(sd)
    wl = FLife.WirschingLight(sd)
    rice = FLife.Rice(sd)
    gm = FLife.GaoMoan(sd)
    pz = FLife.PetrucciZuccarello(sd)

    # Test PDF's; expected result should be 1
    PDFs = {
        'Dirlik': quad(dirlik.get_PDF, a=0, b=np.Inf)[0],
        'Rice': quad(nb.get_PDF, a=0, b=np.Inf)[0],
        'Rice -inf': quad(rice.get_PDF, a=-np.Inf, b=np.Inf)[0],
        'Tovo Benasciutti 1': quad(tb.get_PDF, a=0, b=np.Inf, args=('method 1',))[0],
        'Tovo Benasciutti 2': quad(tb.get_PDF, a=0, b=np.Inf, args=('method 2',))[0],
        'Zhao Baker 1': quad(zb.get_PDF, a=0, b=np.Inf, args=('method 1',))[0],
        'Zhao Baker 2': quad(zb.get_PDF, a=0, b=np.Inf, args=('method 2',))[0],
    }
    for method, value in PDFs.items():
        np.testing.assert_almost_equal(value, 1., decimal=5, err_msg=f'Method: {method}')

    results = {
        'Rainflow': rf.get_life(C = C, k=k, algorithm='four-point'),
        'Rainflow-Goodman': rf.get_life(C = C, k = k, Su=Su),
        'Dirlik': dirlik.get_life(C = C, k=k),
        'Tovo Benasciutti 1': tb.get_life(C = C, k=k, method='method 1'),
        'Tovo Benasciutti 2': tb.get_life(C = C, k=k),
        'Zhao Baker 1': zb.get_life(C = C, k=k),
        'Zhao Baker 2': zb.get_life(C = C, k=k, method='method 2'),
        'Narrowband': nb.get_life(C = C, k=k),
        'Alpha 0.75': a075.get_life(C = C, k=k),
        'Wirsching Light': wl.get_life(C = C, k=k),
        'Rice': rice.get_life(C = C, k=k),
        'Gao Moan': gm.get_life(C = C, k=k),
        'Petrucci Zuccarello': pz.get_life(C = C, k=k, Su=Su)
    }

    for method, value in results.items():
        if method=='Petrucci Zuccarello':
            compare_to = 'Rainflow-Goodman'
        else:
            compare_to = 'Rainflow'
        err = FLife.tools.relative_error(value, results[compare_to])
        print(f'{method:>19s}:{value:6.0f} s,{100*err:>4.0f} % to {compare_to}')
        np.testing.assert_almost_equal(value, results_ref[method], decimal=5, err_msg=f'Method: {method}')

    results_via_PDF = {
        'Dirlik': dirlik.get_life(C = C, k=k, integrate_pdf=True),
        'Tovo Benasciutti 1': tb.get_life(C = C, k=k, method='method 1', integrate_pdf=True),
        'Tovo Benasciutti 2': tb.get_life(C = C, k=k, integrate_pdf=True),
        'Zhao Baker 1': zb.get_life(C = C, k=k, integrate_pdf=True),
        'Zhao Baker 2': zb.get_life(C = C, k=k, method='method 2', integrate_pdf=True),
        'Narrowband': nb.get_life(C = C, k=k, integrate_pdf=True),
        'Alpha 0.75': a075.get_life(C = C, k=k),
        'Wirsching Light': wl.get_life(C = C, k=k),
        'Rice': rice.get_life(C = C, k=k),
        'Gao Moan': gm.get_life(C = C, k=k),
        'Petrucci Zuccarello': pz.get_life(C = C, k=k, Su=Su)
    }

    for method, value in results_via_PDF.items():
        np.testing.assert_almost_equal(value/results[method], 1.0, decimal=2, err_msg=f'Method: {method}')


if __name__ == "__main__":
    test_data()
    #test_version()

if __name__ == '__mains__':
    np.testing.run_module_suite()