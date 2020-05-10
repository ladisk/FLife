import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
import numpy as np


dt = 1e-4
x = np.random.normal(scale=100, size=10000)

C = 1.8e+22
k = 7.3

# Spectral data
sd = FLife.SpectralData(input=x, dt=dt)

# Rainflow reference fatigue life
rf = FLife.Rainflow(sd)

# Spectral methods
dirlik = FLife.Dirlik(sd)
tb = FLife.TovoBenasciutti(sd)
print('          Rainflow: {0:4.0f} s'.format(rf.get_life(C = C, k=k)))
print('            Dirlik: {0:4.0f} s'.format(dirlik.get_life(C = C, k=k)))
print('Tovo Benasciutti 2: {0:4.0f} s'.format(tb.get_life(C = C, k=k, method='improved')))