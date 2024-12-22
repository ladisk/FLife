import numpy as np
from scipy.integrate import quad
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife

def test_version():
    """ check FLife exposes a version attribute """
    assert hasattr(FLife, '__version__')
    assert isinstance(FLife.__version__, str)

def test_data():
    results_ref = {
        'Rainflow': 1399.409574,
        'Rainflow-Low bimodal': 1399.409574,
        'Dirlik': 1595.031603,
        'Tovo Benasciutti 1': 1099.438159,
        'Tovo Benasciutti 2': 1657.218508,
        'Tovo Benasciutti 3': 1842.281870,
        'Zhao Baker 1': 1473.240454,
        'Zhao Baker 2': 1564.902582,
        'Narrowband': 1064.200649,
        'Alpha 0.75': 1625.785759,
        'Wirsching Light': 1461.814034,
        'Gao Moan': 2237.738423,
        'Jiao Moan': 813.490369,
        'Fu Cebon': 1151.170596,
        'Modified Fu Cebon': 846.208325,
        'Low bimodal': 1418.290961,
        'Sakai Okamura': 4966.399683,
        'Bands method': 1969.738539,
        'Single moment': 1969.738539,
        'Ortiz Chen': 1598.836389,
        'Park': 1707.807255,
        'Jun Park': 1681.742086, 
        'Low 2014': 1466.400479,
        'Lotsberg': 476.014139, 
        'Huang Moan': 1439.603712
    }

    data = np.load(os.path.join('./data/m1.npy'))

    t= data[::2]
    x = data[1::2]

    rms = 100  
    C = 1.8e+19
    k = 6
    x = rms * x / np.std(x) 

    # Spectral data
    sd = FLife.SpectralData(input=(x,t[1]), nperseg=int(0.1/t[1]))

    # Rainflow reference fatigue life
    rf = FLife.Rainflow(sd)

    # Spectral methods
    nb = FLife.Narrowband(sd)
    wl = FLife.WirschingLight(sd)
    oc = FLife.OrtizChen(sd)
    a075 = FLife.Alpha075(sd)
    tb = FLife.TovoBenasciutti(sd)
    dk = FLife.Dirlik(sd)
    zb = FLife.ZhaoBaker(sd)
    pk = FLife.Park(sd)
    jp = FLife.JunPark(sd)
    jm = FLife.JiaoMoan(sd)
    so = FLife.SakaiOkamura(sd)
    fc = FLife.FuCebon(sd)
    mfc = FLife.ModifiedFuCebon(sd)
    low_BM = FLife.Low(sd)
    low_2014 = FLife.Low2014(sd)
    gm = FLife.GaoMoan(sd)
    lb = FLife.Lotsberg(sd)
    hm = FLife.HuangMoan(sd)
    sm = FLife.SingleMoment(sd)
    bm = FLife.BandsMethod(sd)

    # Test PDF's; expected result should be 1
    PDFs = {
        'Rice': quad(sd.get_peak_PDF, a=-np.inf, b=np.inf)[0],
        'Tovo Benasciutti 1': quad(tb.get_PDF, a=0, b=np.inf, args=('method 1',))[0],
        'Tovo Benasciutti 2': quad(tb.get_PDF, a=0, b=np.inf, args=('method 2',))[0],
        'Tovo Benasciutti 3': quad(tb.get_PDF, a=0, b=np.inf, args=('method 3',))[0],
        'Dirlik':  quad(dk.get_PDF, a=0, b=np.inf)[0],
        'Zhao Baker 1': quad(zb.get_PDF, a=0, b=np.inf, args=('method 1',))[0],
        'Zhao Baker 2': quad(zb.get_PDF, a=0, b=np.inf, args=('method 2',))[0],
        'Park': quad(pk.get_PDF, a=0, b=np.inf)[0],
        #'Jun Park': quad(jp.get_PDF, a=0, b=np.inf)[0] # Correction factor scales the PDF
    }
    for method, value in PDFs.items():
        np.testing.assert_almost_equal(value, 1., decimal=5, err_msg=f'Method: {method}')

    results = {
        'Rainflow': rf.get_life(C=C, k=k, algorithm='four-point'),
        'Rainflow-Low bimodal': rf.get_life(C=C, k=round(k), algorithm='four-point'),
        'Narrowband': nb.get_life(C=C, k=k),
        'Wirsching Light': wl.get_life(C=C, k=k),
        'Ortiz Chen': oc.get_life(C=C, k=k),
        'Alpha 0.75': a075.get_life(C=C, k=k),
        'Tovo Benasciutti 1': tb.get_life(C=C, k=k, method='method 1'),
        'Tovo Benasciutti 2': tb.get_life(C=C, k=k),
        'Tovo Benasciutti 3': tb.get_life(C=C, k=k, method='method 3'),
        'Dirlik': dk.get_life(C=C, k=k),
        'Zhao Baker 1': zb.get_life(C=C, k=k),
        'Zhao Baker 2': zb.get_life(C=C, k=k, method='method 2'),
        'Park': pk.get_life(C=C, k=k),
        'Jun Park': jp.get_life(C=C, k=k), 
        'Jiao Moan': jm.get_life(C=C, k=k),
        'Sakai Okamura': so.get_life(C=C, k=k),
        'Fu Cebon': fc.get_life(C=C, k=k),
        'Modified Fu Cebon': mfc.get_life(C=C, k=k),
        'Low bimodal': low_BM.get_life(C=C, k=round(k)),
        'Low 2014': low_2014.get_life(C=C, k=k), 
        'Gao Moan': gm.get_life(C=C, k=k),
        'Lotsberg': lb.get_life(C=C, k=k),
        'Huang Moan': hm.get_life(C=C, k=k),
        'Single moment': sm.get_life(C=C, k=k),
        'Bands method': bm.get_life(C=C, k=k)
    }

    for method, value in results.items():
        if method == 'Low bimodal':
            err = FLife.tools.relative_error(value, results['Rainflow-Low bimodal'])
        else:
            err = FLife.tools.relative_error(value, results['Rainflow'])
        print(f'{method:>19s}:{value:6.0f} s,{100*err:>4.0f} % to {"Rainflow"}')
        np.testing.assert_almost_equal(value, results_ref[method], decimal=5, err_msg=f'Method: {method}')

    results_via_PDF = {
        'Narrowband': nb.get_life(C = C, k=k, integrate_pdf=True),
        'Tovo Benasciutti 1': tb.get_life(C = C, k=k, method='method 1', integrate_pdf=True),
        'Tovo Benasciutti 2': tb.get_life(C = C, k=k, integrate_pdf=True),
        'Tovo Benasciutti 3': tb.get_life(C = C, k=k, method='method 3', integrate_pdf=True),
        'Dirlik': dk.get_life(C = C, k=k, integrate_pdf=True),
        'Zhao Baker 1': zb.get_life(C = C, k=k, integrate_pdf=True),
        'Zhao Baker 2': zb.get_life(C = C, k=k, method='method 2', integrate_pdf=True),
        'Park': pk.get_life(C = C, k=k, integrate_pdf=True),
        'Jun Park': jp.get_life(C=C, k=k, integrate_pdf=True) 
    }

    for method, value in results_via_PDF.items():
        np.testing.assert_almost_equal(value/results[method], 1.0, decimal=2, err_msg=f'Method: {method}')


if __name__ == "__main__":
    test_data()
    #test_version()

if __name__ == '__mains__':
    np.testing.run_module_suite()