from __future__ import division

from qubic import QubicScene
from qubic.io import read_map, write_map
from pyspark import SparkContext
from scipy.constants import c, h, k

import numpy as np
import gc

def deriv_and_const(nu, nside, T):
    # Power contribution for each pixel of the sky
    return (8 * np.pi / (12 * nside**2) * 1e-6 * h**2 * nu**4 / 
            (k * c**2 * T**2) * np.exp(h * nu / (k * T)) / 
            (np.expm1(h * nu / (k * T)))**2)

def weighted_beam(beam, nu, nside, T, w):
    return beam * deriv_and_const(nu, nside, T) * w

def linear_weights(g):
    # Weights for linear integration
    w = np.full(len(g), (g[len(g)-1] - g[0]) / (len(g)-1))
    w[0] /= 2
    w[len(g)-1] /= 2
    return w

def non_linear_weights(freq):
    # weights for non linear integration
    w = np.zeros(len(freq))
    w[0] = 0.5 * (freq[1] - freq[0])
    w[1:-1] = 0.5 * (freq[2:]- freq[:-2])
    w[-1] = 0.5 * (freq[-1] - freq[-2])
    return w

def quadratic_grid(f_start, f_stop, n):
    f = np.arange(n)**2
    return f_start + f / f.max() * (f_stop - f_start)

def nearest_freqs(g, f):
    index = np.argmax(g - f > 0)
    f1 = g[index-1]
    f2 = g[index]
    return f1, f2

def read_(nside, nu, idet, theta_max, syb_f=None, NPOINTS=1):
    path = '/home/fincardona/Qubic/spatial_extension/maps/mono'
    if syb_f is None: 
        return read_map(
            '%s/interfero/sb_nside{}_nu{:.3e}_idet{}_tmax{}_Npoints{}.fits'.
            format(nside, nu, idet, theta_max, NPOINTS) % path)
    else: 
        return read_map(
            '%s/direct_conv/dc_nside{}_nu{:.3e}_idet{}_tmax{}_sybf{}_Npoints{}.fits'.format(nside, nu, idet, theta_max, syb_f, 
            NPOINTS) % path)

def write_poly(g, n, beam, nside, idet, theta_max, syb_f=None, NPOINTS=1):
    path = '/home/fincardona/Qubic/spatial_extension/maps/poly'
    if syb_f is None:
        return write_map(
            '%s/interfero/sb_nside{}_poly{}_{}Ghz_idet{}_tmax{}_Npoints{}.fits'.format(nside, int(n), int(np.mean(g)/1e9), idet, theta_max,
            NPOINTS) % path, beam)
    else: 
        return write_map(
            '%s/direct_conv/dc_nside{}_poly{}_{}Ghz_idet{}_tmax{}_sybf{}_Npoints{}.fits'.format(nside, int(n), int(np.mean(g)/1e9), idet, 
            theta_max, syb_f, NPOINTS) % path, beam)

def write_mono(beam, nside, nu, idet, theta_max, syb_f=None, NPOINTS=1):
    path = '/home/fincardona/Qubic/spatial_extension/maps/poly'
    if syb_f is None: 
        return write_map(
            '%s/interfero/sb_nside{}_mono_{}Ghz_idet{}_tmax{}_Npoints{}.fits'
            .format(nside, int(nu/1e9), idet, theta_max, NPOINTS) % path, 
            beam)
    else: 
        return write_map(
            '%s/direct_conv/dc_nside{}_mono_{}Ghz_idet{}_tmax{}_sybf{}_Npoints{}.fits'.format(nside, int(nu/1e9), idet, theta_max, 
            syb_f, NPOINTS) % path, beam)

def linear_integral(g, n, nside, T, idet, theta_max, syb_f=None, NPOINTS=1):
    # Integral on a linear grid
    weights = linear_weights(g)
    sb = np.zeros(12 * nside**2)
    for i, nu in enumerate(g):
        sb_ = read_(nside, nu, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)
        sb += weighted_beam(sb_, nu, nside, T, weights[i])
        gc.collect()
    write_poly(
        g, n, sb, nside, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)

def interpolation(g, f, nside, idet, theta_max, syb_f=None, NPOINTS=1):
    f1, f2 = nearest_freqs(g, f)
    sb1 = read_(nside, f1, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)
    sb2 = read_(nside, f1, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)
    alpha = (f - f1) / (f2 - f1)
    return (1 - alpha) * sb1 + alpha * sb2
    
def non_linear_integral(
        g, n, nside, T, idet, theta_max, syb_f=None, NPOINTS=1):
    f_norm = quadratic_grid(g[0], g[len(g)-1], n)
    weights = non_linear_weights(f_norm)
    sb = np.zeros(12*nside**2)
    for i, nu in enumerate(f_norm):
        sb_ = interpolation(
            g, nu, nside, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)
        sb += weighted_beam(sb_, nu, nside, T, weights[i])
        gc.collect()
    write_poly(
        g, n, sb, nside, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)

def total_integrals(g, n, nside, T, idet, theta_max, syb_f=None, NPOINTS=1):
    band = g[-1] - g[0]
    nu_cent = np.mean(g) 
    if n == 1:
        sb = read_(
            nside, nu_cent, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)
        sb = weighted_beam(sb, nu_cent, nside, T, band)
        write_mono(
            sb, nside, nu_cent, idet, theta_max, syb_f=syb_f, 
            NPOINTS=NPOINTS)
    elif n == len(g):
        linear_integral(
            g, n, nside, T, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)
    else:
        non_linear_integral(
            g, n, nside, T, idet, theta_max, syb_f=syb_f, NPOINTS=NPOINTS)

def write(
        on, off, r, nside, T, idet, theta_max, syb_f=None, NPOINTS=1, 
        kind=True):
    for cut_on, cut_off, res in zip(on, off, r):
        grid = np.linspace(cut_on, cut_off, res)
        sample = np.unique(np.round(np.logspace(0, np.log10(res))))
        for n in sample:
            if kind is True:
                total_integrals(
                    grid, n, nside, T, idet, theta_max, NPOINTS=NPOINTS)
                total_integrals(
                    grid, n, nside, T, idet, theta_max, syb_f=syb_f, 
                    NPOINTS=NPOINTS)

            elif kind == 'interfero':
                total_integrals(
                    grid, n, nside, T, idet, theta_max, NPOINTS=NPOINTS)

            elif kind == 'direct_conv':
                total_integrals(
                    grid, n, nside, T, idet, theta_max, syb_f=syb_f, 
                    NPOINTS=NPOINTS)

            else: print('ERROR')
            gc.collect()

if __name__ == "__main__":

    # Parameters
    nside = 1024
    T_cmb = QubicScene().T_cmb         # Kelvin
    
    idet = 231

    syb_f = 0.99
    theta_max = 30
    
    NPOINTS = [4, 9, 16, 25, 36]
    
    on = [130e9, 190e9]
    off = [170e9, 250e9]
    r = [401, 601]

    # Synthesized Beams
    sc = SparkContext(appName="WriteMultiPolyBeams")
    sc.parallelize(NPOINTS).map(lambda x: write(
        on, off, r, nside, T_cmb, idet, theta_max, syb_f, x)).collect()
    
    sc.stop()
