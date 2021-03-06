from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from pyqt_fit import kde
from scipy.stats import norm

v946 = np.loadtxt('sloan8_yang946.dat', usecols=(30,), unpack=True)
v185 = np.loadtxt('sloan8_yang185.dat', usecols=(30,), unpack=True)

N0 = 1000
N1 = 1000
i = 0
yy946 = np.zeros((N0,N1))
x = np.linspace(-4,4,N1)

while (i < N0):
    print(i)
    v_samp = np.random.choice(v946, size=v946.size, replace=True)
    dist = kde.KDE1D(v_samp)
    yy946[i] = dist(x)
    i += 1

i = 0
yy185 = np.zeros((N0,N1))
x = np.linspace(-4,4,N1)

while (i < N0):
    print(i)
    v_samp = np.random.choice(v185, size=v185.size, replace=True)
    dist = kde.KDE1D(v_samp)
    yy185[i] = dist(x)
    i += 1

y946 = np.median(yy946, axis=0)
y946_up = np.percentile(yy946, 84, axis=0)
y946_down = np.percentile(yy946, 16, axis=0)

y185 = np.median(yy185, axis=0)
y185_up = np.percentile(yy185, 84, axis=0)
y185_down = np.percentile(yy185, 16, axis=0)

#####################################

plt.rc('font', family='serif')

lw=1.7
alph=0.3

fig = plt.figure()
fig.subplots_adjust(wspace=0.25)

ax0 = fig.add_subplot(222)

ax0.set_xlim([-3.2,3.2])
ax0.set_ylim([0,0.42])
ax0.set_xticks(np.linspace(-3,3,7))
ax0.set_yticks(np.linspace(0.1,0.4,4))
ax0.set_xlabel(r'$\Delta c z / \sigma$', fontsize=16)
ax0.set_ylabel(r'$\mathrm{Density}$', fontsize=16)
ax0.tick_params(axis='both', labelsize=12)

ax0.plot(x, y946, color='#e38217', ls='-', lw=lw)
ax0.plot(x, norm.pdf(x), color='#444444', ls='--', lw=lw)
ax0.fill_between(x, y946_up, y946_down, color='#e38217', lw=0, alpha=alph)

ax0.text(-3, 0.335, 'Non-Gaussian \n group', fontsize=11)
ax0.text(2.5, 0.37, '(b)', fontsize=11)

ax1 = fig.add_subplot(221)

ax1.set_xlim([-3.2,3.2])
ax1.set_ylim([0,0.42])
ax1.set_xticks(np.linspace(-3,3,7))
ax1.set_yticks(np.linspace(0.1,0.4,4))
ax1.set_xlabel(r'$\Delta c z / \sigma$', fontsize=16)
ax1.set_ylabel(r'$\mathrm{Density}$', fontsize=16)
ax1.tick_params(axis='both', labelsize=12)

ax1.plot(x, y185, color='#e38217', ls='-', lw=lw)
ax1.plot(x, norm.pdf(x), color='#444444', ls='--', lw=lw)
ax1.fill_between(x, y185_up, y185_down, color='#e38217', lw=0, alpha=alph)

ax1.text(-3, 0.335, 'Gaussian \n group', fontsize=11)
ax1.text(2.5, 0.37, '(a)', fontsize=11)
#ax1.text(-1, 0.1, 'Gaussian', fontsize=10)

fig.savefig('vdist.pdf', dpi=1200, bbox_inches='tight')
