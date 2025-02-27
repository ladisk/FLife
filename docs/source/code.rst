Code documentation
==================

Spectral data
-------------
.. autoclass:: FLife.SpectralData

Frequency-domain fatigue life
-----------------------------
Narrowband
**********
.. autoclass:: FLife.freq_domain.Narrowband

Wirsching-Light
***************
.. autoclass:: FLife.freq_domain.WirschingLight
    :exclude-members: get_PDF

Ortiz-Chen
**********
.. autoclass:: FLife.freq_domain.OrtizChen
    :exclude-members: get_PDF

Alpha075
********
.. autoclass:: FLife.freq_domain.Alpha075
    :exclude-members: get_PDF

Tovo-Benasciutti
****************
.. autoclass:: FLife.freq_domain.TovoBenasciutti

Dirlik
******
.. autoclass:: FLife.freq_domain.Dirlik

Zhao-Baker
**********
.. autoclass:: FLife.freq_domain.ZhaoBaker

Park
****
.. autoclass:: FLife.freq_domain.Park

Jun-Park
********
.. autoclass:: FLife.freq_domain.JunPark

Jiao-Moan
*********
.. autoclass:: FLife.freq_domain.JiaoMoan
    :exclude-members: get_PDF

Sakai-Okamura
*************
.. autoclass:: FLife.freq_domain.SakaiOkamura
    :exclude-members: get_PDF

Fu-Cebon
********
.. autoclass:: FLife.freq_domain.FuCebon
    :exclude-members: get_PDF

Modified Fu-Cebon
*****************
.. autoclass:: FLife.freq_domain.ModifiedFuCebon
    :exclude-members: get_PDF

Low's bimodal
*************
.. autoclass:: FLife.freq_domain.Low

Low 2014
**********
.. autoclass:: FLife.freq_domain.Low2014
    :exclude-members: get_PDF

Lotsberg
********
.. autoclass:: FLife.freq_domain.Lotsberg
    :exclude-members: get_PDF   

Huang-Moan
**********
.. autoclass:: FLife.freq_domain.HuangMoan
    :exclude-members: get_PDF 

Gao-Moan
********
.. autoclass:: FLife.freq_domain.GaoMoan
    :exclude-members: get_PDF

Single moment
*************
.. autoclass:: FLife.freq_domain.SingleMoment
    :exclude-members: get_PDF

Bands method
*************
.. autoclass:: FLife.freq_domain.BandsMethod
    :exclude-members: get_PDF

Time-domain fatigue life
------------------------
.. autoclass:: FLife.time_domain.Rainflow

Multiaxial fatigue life
-----------------------
.. autoclass:: FLife.multiaxial.EquivalentStress

Visualize
*********
.. autofunction:: FLife.visualize.pick_point