from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from pyqt_fit import kde
from scipy.optimize import curve_fit

gr = np.loadtxt('gr_z05.dat', unpack=True)

x = np.linspace(0, 1.1, 500)
xx = np.linspace(0, 1.1, 100)
dist = kde.KDE1D(gr)
y = dist(x)
yy = dist(xx)

def norm(x, mu, sig):
    return (1 / (sig * np.sqrt(np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def f(x, a0, mu0, sig0, a1, mu1, sig1):
    return a0 * norm(x, mu0, sig0) + a1 * norm(x, mu1, sig1)

fit = curve_fit(f, x, y, p0 = [1,0.6,0.2,3,0.8,0.1])

print(fit[0])

#####################################

plt.rc('font', family='serif')

lw = 1
mew = 1
ms = 6

blue = '#0d4f8b'
red = '#b22222'

fig = plt.figure()

ax0 = fig.add_subplot(221)

ax0.set_xlim([0.1,1.1])
ax0.set_ylim([0,2.2])
ax0.set_xticks(np.linspace(0.2,1,5))
ax0.set_yticks(np.linspace(0.5,2,4))
ax0.set_xlabel('g - r', fontsize=11)
ax0.set_ylabel('Density', fontsize=11)
ax0.tick_params(axis='both', labelsize=9)

ax0.plot(xx, yy, mec='#666666', mfc='none', marker='o', ms=ms, mew=mew, ls='none')
ax0.plot(x, f(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5]), 'k-', lw=lw+1)
ax0.plot(x, fit[0][0] * norm(x,fit[0][1],fit[0][2]), color=blue, ls='-', lw=lw)
ax0.plot(x, fit[0][3] * norm(x,fit[0][4],fit[0][5]), color=red, ls='-', lw=lw)

ax0.text(0.15, 2, 'Blue sequence + red cloud', fontsize=9)
ax0.text(0.15, 1.85, 'Blue sequence', color=blue, fontsize=9)
ax0.text(0.15, 1.7, 'Red cloud', color=red, fontsize=9)

fig.savefig('gr_dist.pdf', dpi=1200, bbox_inches='tight')
