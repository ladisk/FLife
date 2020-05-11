import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
import lvm_read
import numpy as np

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

C = 1.8e+22
k = 7.3
Su = 446

results = {
    'Rainflow': rf.get_life(C = C, k=k),
    'Rainflow-Goodman': rf.get_life(C = C, k = k, Su=Su),
    'Dirlik': dirlik.get_life(C = C, k=k),
    'Tovo Benasciutti 1': tb.get_life(C = C, k=k, method='method 1'),
    'Tovo Benasciutti 2': tb.get_life(C = C, k=k, method='method 2'),
    'Zhao Baker 1': zb.get_life(C = C, k=k, method='method 1'),
    'Zhao Baker 2': zb.get_life(C = C, k=k, method='method 2'),
    'Narrowband': nb.get_life(C = C, k=k),
    'Alpha 0.75': a075.get_life(C = C, k=k),
    'Wirsching Light': wl.get_life(C = C, k=k),
    'Rice': rice.get_life(C = C, k=k),
    'Gao Moan': gm.get_life(C = C, k=k),
    'Petrucci Zuccarello': pz.get_life(C = C, k=k, Su=Su)
}

for k, v in results.items():
    if k=='Petrucci Zuccarello':
        compare_to = 'Rainflow-Goodman'
    else:
        compare_to = 'Rainflow'
    err = FLife.tools.relative_error(v, results[compare_to])
    print(f'{k:>19s}:{v:6.0f} s,{100*err:>4.0f} % to {compare_to}')