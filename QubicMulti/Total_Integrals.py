from __future__ import division

from qubic import *
from qubic.io import (read_map, write_map)
from qubic.utils import progress_bar
from scipy.constants import c, h, k

import numpy as np
import qubic

def deriv(nu, nside, T):
    # Power contribution for each pixel of the sky
    der = (8*np.pi/(12*nside**2)*1e-6*h**2*nu**4/(k*c**2*T**2)*
          np.exp(h*nu/(k*T))/(np.expm1(h*nu/(k*T)))**2)
    return der

def linear_weights(g):
    # Weights for linear integration
    w = np.full(len(g), (g[len(g)-1] - g[0])/(len(g)-1))
    w[0] /= 2
    w[len(g)-1] /= 2
    return w

def linear_integral(g, nside, T):
    # Integral on a linear grid
    weights = linear_weights(g)
    sb = np.zeros(12*nside**2)
    bar = progress_bar(len(g))
    for i, nu in enumerate(g):
        sb_ = read_map('sb_nside{}_nu{:.3e}.fits'.format(nside, nu))
        sb += sb_*deriv(nu, nside, T)*weights[i]
        bar.update()
    write_map('sb_nside{}_poly{}_{}Ghz.fits'.format(
        nside, len(g), int(np.mean(grid)/1e9)), sb)

def quadratic_grid(f_start, f_stop, n):
    f = np.arange(n)**2
    return f_start + f / f.max() * (f_stop - f_start)

def non_linear_weights(freq):
    # weights for non linear integration
    w = np.zeros(len(freq))
    w[0] = 0.5 * (freq[1] - freq[0])
    w[1:-1] = 0.5 * (freq[2:]- freq[:-2])
    w[-1] = 0.5 * (freq[-1] - freq[-2])
    return w

def near_freq(g, f):
    index = np.argmax(g - f > 0)
    f1 = g[index-1]
    f2 = g[index]
    return f1, f2

def interpolation(g, f, nside):
    f1, f2 = near_freq(g, f)
    sb1 = read_map('sb_nside{}_nu{:.3e}.fits'.format(nside, f1))
    sb2 = read_map('sb_nside{}_nu{:.3e}.fits'.format(nside, f2))
    alpha = (f - f1)/(f2 - f1)
    y = (1 - alpha)*sb1 + alpha*sb2
    return y

def non_linear_integral(g, n, nside, T):
    f_norm = quadratic_grid(g[0], g[len(g)-1], n)
    weights = non_linear_weights(f_norm)
    bar = progress_bar(n)
    sb = np.zeros(12*nside**2)
    for i, f in enumerate(f_norm):
        sb_ = interpolation(g, f, nside)
        sb += sb_*deriv(f, nside, T)*weights[i]
        bar.update()
    write_map('sb_nside{}_poly{}_{}Ghz.fits'.format(
        nside, int(n), int(np.mean(grid)/1e9)), sb)


N = [512,  1024, 2048]
T_cmb = 2.7255          # Kelvin

nu_cent = 150e9
cut_on = 130e9          # Ideal Filter 150 GHz - 25% bandwidht
cut_off = 170e9         
res = 401

#nu_cent = 220e9
#cut_on = 190e9         # Ideal Filter 220 GHz - 25% bandwidht
#cut_off = 250e9        
#res = 601

grid = np.linspace(cut_on, cut_off, res)
sample = np.unique(np.round(np.logspace(0, np.log10(res))))

for nside in N:
    bar = progress_bar(len(sample))
    for n in sample:
        if n == 1:
            sb = read_map('sb_nside{}_nu{:.3e}.fits'.format(nside, nu_cent))
            sb *= deriv(nu_cent, nside, T_cmb)*(cut_off - cut_on)
            write_map('sb_nside{}_mono_{}Ghz.fits'.format(nside, int(nu_cent/1e9)), 
                      sb)
        elif n == res:
            linear_integral(grid, nside, T_cmb)
        else:
            non_linear_integral(grid, n, nside, T_cmb)
            bar.update()
