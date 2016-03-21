from __future__ import division

from qubic import plot_spectra, Xpol

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import glob

def cl2dl(ell, Cl):
    return Cl * ell * (ell + 1) / 2 / np.pi 

NFREQS = 1
NPOINTS = 1

nside = 256

nsample = 1000
maxiter = 1000

path = '/home/fincardona/Qubic/map_making/maps/montecarlo'
cov = hp.read_map('%s/qcoverage_n{}_N{}_s{}_mi{}.fits'.format(
    NFREQS, NPOINTS, nsample, maxiter) % path)
maskmap = cov > cov.max() * 0.2 

lmin = 2
lmax = nside * 2  
delta_ell = 20 

xpol = Xpol(maskmap, lmin, lmax, delta_ell) 
ell_binned = xpol.ell_binned
nbins = len(ell_binned)

reconstructed_maps = glob.glob('%s/fusion_n{}_N{}_s{}_mi{}*'.format(
    NFREQS, NPOINTS, nsample, maxiter) % path)
all_spectra = np.empty((len(reconstructed_maps), 6, nbins))
for i, f in enumerate(reconstructed_maps):
    rec_map = hp.read_map(f, field=(0,1,2))
    biased, unbiased = xpol.get_spectra(rec_map) 
    all_spectra[i] = unbiased 

Cls = all_spectra.mean(axis=0)
delta_Cls = all_spectra.std(axis=0)

# and now we need to multiply it by the beam. For gaussian beam:
beam = lambda x: np.exp( -x * (x + 1) * 0.5 * np.radians(0.39/2.35)**2)**2

Cls /= beam(ell_binned)
delta_Cls /= beam(ell_binned)
np.save('./Xpol/delta_Cls_n{}_N{}_bin{}_{}MAPS_s{}_mi{}'.format(
    NFREQS, NPOINTS, delta_ell, len(reconstructed_maps), nsample, 
    maxiter), [ell_binned, delta_Cls])

Dls = cl2dl(ell_binned, Cls)
delta_Dls = cl2dl(ell_binned, delta_Cls)

mp.figure()
mp.loglog(ell_binned, Dls[2], label='$D_\ell^{BB}\ $')
mp.plot(ell_binned, delta_Dls[2], label='$ \Delta D_\ell^{BB}\ $' +
        'n{} N{}'.format(NFREQS, NPOINTS))
mp.xlim(40, 300)
#mp.ylim([1e-4, 1e-1])
mp.xlabel('$\ell$', fontsize=16)
mp.legend(loc='best')
mp.show()

