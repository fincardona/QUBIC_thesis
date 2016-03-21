from __future__ import division

from qubic.io import read_map
from qubic import read_spectra

import random
import numpy as np
import healpy as hp
import matplotlib.pyplot as mp

def read_poly(nside, n, nu, idet, theta_max, syb_f=None):
    path = '/home/federico/Desktop/MAPS/no_spat_ext/poly'
    if syb_f is None and n==1:
        return read_map(
            '%s/interfero/sb_nside{}_mono_{}Ghz_idet{}_tmax{}.fits'.
            format(nside, int(nu/1e9), idet, theta_max) % path)
    elif syb_f is None: 
        return read_map(
            '%s/interfero/sb_nside{}_poly{}_{}Ghz_idet{}_tmax{}.fits'.
            format(nside, int(n), int(nu/1e9), idet, theta_max) % path)
    elif syb_f is not None and n==1:
        return read_map(
            '%s/direct_conv/dc_nside{}_mono_{}Ghz_idet{}_tmax{}_sybf{}.fits'.
            format(nside, int(nu/1e9), idet, theta_max, syb_f) % path)
    elif syb_f is not None:
        return read_map(
            '%s/direct_conv/dc_nside{}_poly{}_{}Ghz_idet{}_tmax{}_sybf{}.fits'.
            format(nside, int(n), int(nu/1e9), idet, theta_max, syb_f) % path)
    else: 
        raise ValueError('Error')

def read_poly_N(nside, n, nu, idet, theta_max, syb_f=None, NPOINTS=1):
    path = '/home/federico/Desktop/MAPS/spat_ext/poly'
    if syb_f is None and n==1:
        return read_map(
            '%s/interfero/sb_nside{}_mono_{}Ghz_idet{}_tmax{}_Npoints{}.fits'.format(nside, int(nu/1e9), idet, theta_max, NPOINTS) % path)
    elif syb_f is None: 
        return read_map(
            '%s/interfero/sb_nside{}_poly{}_{}Ghz_idet{}_tmax{}_Npoints{}.fits'.format(nside, int(n), int(nu/1e9), idet, theta_max, NPOINTS) % path)
    elif syb_f is not None and n==1:
        return read_map(
            '%s/direct_conv/dc_nside{}_mono_{}Ghz_idet{}_tmax{}_sybf{}_Npoints{}.fits'.format(nside, int(nu/1e9), idet, theta_max, syb_f, NPOINTS) % path)
    elif syb_f is not None:
        return read_map(
            '%s/direct_conv/dc_nside{}_poly{}_{}Ghz_idet{}_tmax{}_sybf{}_Npoints{}.fits'.format(nside, int(n), int(nu/1e9), idet, theta_max, syb_f, NPOINTS) % path)
    else: 
        raise ValueError('Error')

def read_(nside, n, nu_cent, idet, theta_max, syb_f=None, NPOINTS=1):
    if NPOINTS == 1:
        return read_poly(nside, n, nu_cent, idet, theta_max, syb_f=syb_f)
    return read_poly_N(nside, n, nu_cent, idet, theta_max, syb_f=syb_f, 
            NPOINTS=NPOINTS)

def w_function(
        lmax, nside, n, nu_cent, idet, theta_max, syb_f=None, NPOINTS=1, 
        pol=False):
    cls = np.zeros((len(n), len(NPOINTS), lmax))
    ell = np.arange(lmax)
    for i, enn in enumerate(n):
        for j, N in enumerate(NPOINTS):
            beam = read_(
                nside, enn, nu_cent, idet, theta_max, syb_f=syb_f, NPOINTS=N)
            cls[i, j] = hp.anafast(beam, lmax=lmax-1, pol=pol) * (2*ell+1)
    return cls, cls.max() 

def plot(what, freqs, points, n, NPOINTS, title=None):
    mp.figure()
    f_index, p_index = ([], [])
    for f in freqs: 
        f_index.append(n.index(f))
    for p in points:
        p_index.append(NPOINTS.index(p))
    color = ['g', 'r', 'c', 'b', 'y', 'm', 'k']
    style = ['-', '--', '-.', ':']
    for d, i in enumerate(f_index):
        a = color[d]
        for c, j in enumerate(p_index):
            b = style[c]
            if title:
                mp.title('Window Functions - {}'.format(title), fontsize='x-large')
            mp.plot(
                ell, what[i, j, :], a+b,label='n = {}, '.format(
                n[i]) + '$ N_{P} $' + ' = {}'.format(NPOINTS[j]))
            mp.xlabel('$\ell$', fontsize = 'xx-large')
            mp.ylabel('$W_{\ell}$', fontsize = 'xx-large') #W_l = (2l+1)C_l
            mp.xticks(fontsize = 'x-large')
            mp.yticks(fontsize = 'x-large')
    mp.legend(bbox_to_anchor=(1.13, 1.14), fontsize = 'large')

def plot_ratio(what_num, what_den, freqs, points, n, NPOINTS):
    what = what_num/what_den
    plot(what, freqs, points, n, NPOINTS)
    mp.title('Window functions ratio', fontsize='x-large', loc='left')
    mp.ylabel(
        r'${W_{\ell}^{app}}/{W_{\ell}^{int}}$',
        fontsize = 'xx-large', rotation='horizontal',
        horizontalalignment = u'right')
    mp.ylim((0.5, 1.5))
    mp.xticks(fontsize = 'x-large')
    mp.yticks(fontsize = 'x-large')
    mp.legend(bbox_to_anchor=(0.5, 1.14), fontsize = 'large')#bbox_to_anchor=(0.35, 0.6))

nside = 1024
theta_max = 30
syb_f = 0.99
idet = 231
lmax = 500
nu_cent = [150e9, 220e9]
n_150, n_220 = ([1, 3, 19, 401], [1, 9, 23, 601])
NPOINTS = [1, 4, 16, 36]   #NPOINTS = [1, 4, 9, 16, 25, 36]

pol=False

ell = np.arange(lmax)
int_cls150, int_max = w_function(
    lmax, nside, n_150, nu_cent[0], idet, theta_max, syb_f=syb_f, 
    NPOINTS=NPOINTS, pol=pol)

dir_cls150, dir_max = w_function(
    lmax, nside, n_150, nu_cent[0], idet, theta_max, NPOINTS=NPOINTS, 
    pol=pol)

freqs = [1, 3, 19, 401]
points = [1, 4, 16, 36]

plot(int_cls150 / int_max, freqs, points, n_150, NPOINTS, 
     'Interferometric beams')
plot(dir_cls150 / dir_max, freqs, points, n_150, NPOINTS, 
     'GaussianApproximated beams')
plot_ratio(dir_cls150, int_cls150, freqs, points, n_150, NPOINTS)

mp.show()
