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
        'Rainflow': 1399.409574,
        'Rainflow-Goodman': 1300.605740,
        'Dirlik': 1595.031603,
        'Tovo Benasciutti 1': 1099.438159,
        'Tovo Benasciutti 2': 1657.218508,
        'Zhao Baker 1': 1473.240454,
        'Zhao Baker 2': 1564.902582,
        'Narrowband': 1064.200649,
        'Alpha 0.75': 1625.785759,
        'Wirsching Light': 1461.814034,
        'Gao Moan': 1310.451362,
        'Jiao Moan': 813.490369,
        'Fu Cebon': 1151.170596,
        'Modified Fu Cebon': 846.208325,
        'Low': 1418.290961,
        'Sakai Okamura': 4966.399683,
        'Bands method': 1969.738539,
        'Single moment': 1969.738539,
        'Ortiz Chen': 1598.836389,
        'Park': 1707.807255,
        'Jun Park': 1681.742086, 
        'Low bimodal 2014': 1466.400479,
        'Lotsberg': 476.014139, 
        'Huang Moan': 1439.603712
    }

    data = lvm_read.read('./data/m1.lvm')[0]

    t = data['data'][:,0]
    x = data['data'][:,1]

    rms = 100  
    C = 1.8e+19
    k = 6
    Su = 446
    x = rms * x / np.std(x) 

    # Spectral data
    sd = FLife.SpectralData(input=(x,t[1]), nperseg=int(0.1/t[1]))

    # Rainflow reference fatigue life
    rf = FLife.Rainflow(sd)

    # Spectral methods
    dirlik = FLife.Dirlik(sd)
    tb = FLife.TovoBenasciutti(sd)
    zb = FLife.ZhaoBaker(sd)
    nb = FLife.Narrowband(sd)
    a075 = FLife.Alpha075(sd)
    wl = FLife.WirschingLight(sd)
    gm = FLife.GaoMoan(sd)
    jm = FLife.JiaoMoan(sd)
    fc = FLife.FuCebon(sd)
    mfc = FLife.ModifiedFuCebon(sd)
    low = FLife.Low(sd)
    so = FLife.SakaiOkamura(sd)
    bm = FLife.BandsMethod(sd)
    sm = FLife.SingleMoment(sd)
    oc = FLife.OrtizChen(sd)
    park = FLife.Park(sd)
    jp = FLife.JunPark(sd)
    low2014 = FLife.LowBimodal2014(sd)
    lb = FLife.Lotsberg(sd)
    hm = FLife.HuangMoan(sd)

    # Test PDF's; expected result should be 1
    PDFs = {
        'Rice': quad(sd.get_peak_PDF, a=-np.Inf, b=np.Inf)[0],
        'Tovo Benasciutti 1': quad(tb.get_PDF, a=0, b=np.Inf, args=('method 1',))[0],
        'Tovo Benasciutti 2': quad(tb.get_PDF, a=0, b=np.Inf, args=('method 2',))[0],
        'Zhao Baker 1': quad(zb.get_PDF, a=0, b=np.Inf, args=('method 1',))[0],
        'Zhao Baker 2': quad(zb.get_PDF, a=0, b=np.Inf, args=('method 2',))[0],
        'Park': quad(park.get_PDF, a=0, b=np.Inf)[0]
        #'Jun Park': quad(jp.get_PDF, a=0, b=np.Inf)[0]
    }
    for method, value in PDFs.items():
        np.testing.assert_almost_equal(value, 1., decimal=5, err_msg=f'Method: {method}')

    results = {
        'Rainflow': rf.get_life(C=C, k=k, algorithm='four-point'),
        'Rainflow-Goodman': rf.get_life(C=C, k = k, Su=Su),
        'Dirlik': float(dirlik.get_life(C=C, k=k)),
        'Tovo Benasciutti 1': float(tb.get_life(C=C, k=k, method='method 1')),
        'Tovo Benasciutti 2': float(tb.get_life(C=C, k=k)),
        'Zhao Baker 1': zb.get_life(C=C, k=k),
        'Zhao Baker 2': zb.get_life(C=C, k=k, method='method 2'),
        'Narrowband': nb.get_life(C=C, k=k),
        'Alpha 0.75': a075.get_life(C=C, k=k),
        'Wirsching Light': wl.get_life(C=C, k=k),
        'Gao Moan': gm.get_life(C=C, k=k),
        'Jiao Moan': jm.get_life(C=C, k=k),
        'Fu Cebon': fc.get_life(C=C, k=k),
        'Modified Fu Cebon': mfc.get_life(C=C, k=k),
        'Low': low.get_life(C=C, k=int(k)),
        'Sakai Okamura': so.get_life(C=C, k=k),
        'Bands method': bm.get_life(C=C, k=k),
        'Single moment': sm.get_life(C=C, k=k),
        'Ortiz Chen': oc.get_life(C=C, k=k),
        'Park': park.get_life(C=C, k=k),
        'Jun Park': jp.get_life(C=C, k=k), 
        'Low bimodal 2014': low2014.get_life(C=C, k=k), 
        'Lotsberg': lb.get_life(C=C, k=k), 
        'Huang Moan': hm.get_life(C=C, k=k)
    }

    for method, value in results.items():
        err = FLife.tools.relative_error(value, results['Rainflow'])
        print(f'{method:>19s}:{value:6.0f} s,{100*err:>4.0f} % to {"Rainflow"}')
        np.testing.assert_almost_equal(value, results_ref[method], decimal=5, err_msg=f'Method: {method}')

    results_via_PDF = {
        'Dirlik': dirlik.get_life(C = C, k=k, integrate_pdf=True),
        'Tovo Benasciutti 1': tb.get_life(C = C, k=k, method='method 1', integrate_pdf=True),
        'Tovo Benasciutti 2': tb.get_life(C = C, k=k, integrate_pdf=True),
        'Zhao Baker 1': zb.get_life(C = C, k=k, integrate_pdf=True),
        'Zhao Baker 2': zb.get_life(C = C, k=k, method='method 2', integrate_pdf=True),
        'Narrowband': nb.get_life(C = C, k=k, integrate_pdf=True),
        'Park': park.get_life(C = C, k=k, integrate_pdf=True),
        'Jun Park': jp.get_life(C=C, k=k, integrate_pdf=True) 
    }

    for method, value in results_via_PDF.items():
        np.testing.assert_almost_equal(value/results[method], 1.0, decimal=2, err_msg=f'Method: {method}')


if __name__ == "__main__":
    test_data()
    #test_version()

if __name__ == '__mains__':
    np.testing.run_module_suite()