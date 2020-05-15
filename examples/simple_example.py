import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import FLife
import lvm_read
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si


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


#Define stress vector
s = np.arange(0,1000,.001)
#Rice combines Rayleighh and standard Gaussian function, so negative stress amplitudes
#should be considered also, for pdf only
s_rice  = np.arange(-500,1000,.001)

#Get PDF functions
dirlik_pdf = dirlik.get_PDF(s)
TB1_pdf = tb.get_PDF(s, method='method 1')
TB2_pdf = tb.get_PDF(s, method='method 2')
NB_pdf = nb.get_PDF(s)
ZB1_pdf = zb.get_PDF(s, method='method 1')
ZB2_pdf = zb.get_PDF(s, method='method 2')
rice_pdf = rice.get_PDF(s_rice)


plt.plot(s,dirlik_pdf, label='Dirlik')
plt.plot(s,TB1_pdf, label='TB1')
plt.plot(s,TB2_pdf, label='TB2')
plt.plot(s,NB_pdf, label='NB')
plt.plot(s,ZB1_pdf, label='ZB1')
plt.plot(s,ZB2_pdf, label='ZB2')
plt.plot(s_rice,rice_pdf, label='Rice')
plt.legend()

#Preverimo integral PDF-a
si.quad(dirlik._function_PDF(), 0, np.inf)[0]
si.quad(nb._function_PDF(), 0, np.Inf)[0]
si.quad(rice._function_PDF(), -np.Inf, np.Inf)[0]
si.quad(tb._function_PDF(method='method 1'), 0, np.Inf)[0]
si.quad(tb._function_PDF(method='method 2'), 0, np.Inf)[0]
si.quad(zb._function_PDF(method='method 1'), 0, np.Inf)[0]
si.quad(zb._function_PDF(method='method 1'), 0, np.Inf)[0]


#Preverimo pravilnost PDF-a
1/(sd.m_p / C * si.quad(dirlik._function_PDF(k=k),0,np.Inf)[0])
1/(sd.nu / C *  si.quad(tb._function_PDF(method='method 1', k=k),0,np.Inf)[0])
1/(sd.nu / C *  si.quad(tb._function_PDF(method='method 2', k=k),0,np.Inf)[0])
1/(sd.m_p / C *  si.quad(zb._function_PDF(method='method 1', k=k),0,np.Inf)[0])
1/(sd.m_p / C *  si.quad(zb._function_PDF(method='method 2', k=k),0,np.Inf)[0])
1/(sd.nu / C *  si.quad(nb._function_PDF(k=k),0,np.Inf)[0])
1/(sd.m_p / C *  si.quad(rice._function_PDF(k=k),0,np.Inf)[0])


results = {
    'Rainflow': rf.get_life(C = C, k=k),
    'Rainflow-Goodman': rf.get_life(C = C, k = k, Su=Su),
    'Dirlik': dirlik.get_life(C = C, k=k),
    'Tovo Benasciutti 1': tb.get_life(C = C, k=k, method='method 1'),
    'Tovo Benasciutti 2': tb.get_life(C = C, k=k, method='method 2'),
    'Zhao Baker 1': zb.get_life(C = C, k=k ,method='method 1'),
    'Zhao Baker 2': zb.get_life(C = C, k=k, method='method 2'),
    'Narrowband': nb.get_life(C = C, k=k),
    'Alpha 0.75': a075.get_life(C = C, k=k),
    'Wirsching Light': wl.get_life(C = C, k=k),
    'Rice': rice.get_life(C = C, k=k),
    'Gao Moan': gm.get_life(C = C, k=k),
    'Petrucci Zuccarello': pz.get_life(C = C, k=k, Su=Su)
}

for k1, v in results.items():
    if k1=='Petrucci Zuccarello':
        compare_to = 'Rainflow-Goodman'
    else:
        compare_to = 'Rainflow'
    err = FLife.tools.relative_error(v, results[compare_to])
    print(f'{k1:>19s}:{v:6.0f} s,{100*err:>4.0f} % to {compare_to}')

#izris PSD-ja
plt.figure(figsize=(10,5))
plt.plot(sd.psd[:,0],sd.psd[:,1])
plt.xlim(0,1200)

#Dolocimo zgornjo mejo frekvencnega pasu/-ov
#gm = FLife.GaoMoan(sd) #default:  enaka varianca pasov
gm = FLife.GaoMoan(sd, band_frequency=[300,1200])
#gm = FLife.GaoMoan(sd, band_frequency=[1200])
#gm = FLife.GaoMoan(sd, band_frequency=[150.,300.,1200])

results['Gao Moan'] = gm.get_life(C = C, k=k)
print()
for k1, v in results.items():
    if k1=='Petrucci Zuccarello':
        compare_to = 'Rainflow-Goodman'
    else:
        compare_to = 'Rainflow'
    err = FLife.tools.relative_error(v, results[compare_to])
    print(f'{k1:>19s}:{v:6.0f} s,{100*err:>4.0f} % to {compare_to}')