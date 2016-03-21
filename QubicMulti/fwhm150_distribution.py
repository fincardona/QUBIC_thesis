#coding: utf8
from __future__ import division

import matplotlib.pyplot as mp
import matplotlib.mlab as mlab

import numpy as np
import scipy as sp

from qubic.utils import progress_bar

def load_fwhm(nside, det, syb_f, t_max):
    return np.load(
        'fwhm150s/fwhm150_nside{}_det{}_sybf{}_thetamax{}.npy'.format(
        nside, det, syb_f, t_max))

def get_central(bins):
    b = np.diff(bins)       # gap 
    f = bins[:-1] + b/2     # get the central value
    return f

def gaussian(mu, sigma, x=None, normed=True):
    if x is None:
        x = np.linspace(mu - (5*sigma), mu + (5*sigma), 1000)
    if normed is not True:
        return np.exp(-(x - mu)**2/(2*sigma**2))
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2/(2*sigma**2))

nside = 1024  

nu = 150e9     #fixed
syb_f = 1      #fixed

t_max = [30, 60]

det = np.arange(0, 992)
fwhm150 = np.zeros((992, len(t_max), 2))
for i in det:
    fwhm150[i] = load_fwhm(nside, i, syb_f, t_max)

np.save('fwhm150s/fwhm150_nside{}_alldet_sybf{}_thetamax{}'.format(
    nside, syb_f, t_max), fwhm150)

''''
legend: fwhm150   
fwhm150[i, j, k]

i = idet
j = theta_max in enumerate(t_max)
k = 0 : Energy test
k = 1 : Residuals test
'''''

let = ['Energy', 'Residuals']
Nbins = 12

mu_sigma = np.zeros((len(t_max), 2, 2))
for i in range(len(t_max)):
    for l, (j, fwhm) in zip(let, enumerate(
            [fwhm150[:, i, 0], fwhm150[:, i, 1]])):
        mu_sigma[i, j] = sp.stats.norm.fit(fwhm)
        mp.figure(1)
        ax = mp.subplot(len(t_max), 2, 2*i+j+1)
        num, bins, patches = ax.hist(fwhm, bins=Nbins, normed=False)
        y = gaussian(mu_sigma[i, j, 0], mu_sigma[i, j, 1], x=bins, 
                     normed=False)*np.max(num)
        ax.set_xlim(
            np.min(bins)-np.min(bins)*0.000002*(200*j+1), 
            np.max(bins)+np.max(bins)*0.000002*(200*j+1))
        central_indices = np.arange(0, len(bins)-1, 2)
        bin_centers = get_central(bins)[central_indices]
        ax.set_xticks(bin_centers)
        ax.set_xticklabels([str('%.6f') % bin_centers[k] for k in 
                            range(len(central_indices))])
        ax.plot(bins, y, 'r', linewidth=2)
        ax.set_title(r'fwhm@150GHz - {} test - $\theta max$ = {}'.format(
            l, t_max[i]))
        ax.set_xlabel('fwhm')
        mp.figure(i+2)
        num2, bins2, patches2 = mp.hist(fwhm, bins=Nbins, normed=False)
        mp.title('fwhm@150GHz')
        mp.plot(bins2, y, 'r', linewidth=2)
        mp.xlim(0.377, 0.387)

mp.show()

print mu_sigma
