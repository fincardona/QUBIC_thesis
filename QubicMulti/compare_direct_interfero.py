#coding: utf8
from __future__ import division

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

from pyoperators import DenseBlockDiagonalOperator
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator, SceneHealpixCMB)
from qubic import QubicInstrument, QubicScene
from pysimulators.geometry import rotate
from MyQubicInstrument import MyQubicInstrument

nside = 1024
scene = SceneHealpixCMB(nside, kind='I')

nu = 150e9

idet = 100
theta_max = 30
syb_f = 0.99

fwhm150 = 0.385988277  # mettere il valore esatto

q = MyQubicInstrument(
    filter_nu=nu, filter_relative_bandwidth=1/nu, 
    synthbeam_dtype=float, synthbeam_fraction=syb_f, 
    synthbeam_peak150_fwhm=np.radians(fwhm150))

# Interferometry synthetic beam
syb_ref = q.get_synthbeam(scene, idet, theta_max)

# Gaussian approximation
syb_ga = q.direct_convolution(scene, idet, theta_max)

# Display
mp.figure()
ref = hp.gnomview(
    syb_ref, rot=[0, 90], reso=5, xsize=600, min=0, max=0.016, 
    return_projected_map=True, sub=(1, 3, 1), title='Interferometry', 
    margins=4 * [0.01])
ga = hp.gnomview(
    syb_ga, rot=[0, 90], reso=5, xsize=600, min=0, max=0.016,
    return_projected_map=True, sub=(1, 3, 2), title='Gaussian approximation',
    margins=4 * [0.01])
diff = hp.gnomview(
    np.abs(syb_ref-syb_ga), rot=[0, 90], reso=5, xsize=600, min=0, max=0.016, 
    return_projected_map=True, sub=(1, 3, 3), title='Difference',
    margins=4 * [0.01])

hp.gnomview(np.log(syb_ref), rot=[0, 90], reso=10, xsize=600)
hp.gnomview(np.log(syb_ga), rot=[0, 90], reso=10, xsize=600)

i, j = np.unravel_index(np.argmax(ref), ref.shape)
x = np.arange(600) * 5 / 60
x -= x[j]
mp.figure()
mp.plot(x, ref[i], 'g', label='Interferometry')
mp.plot(x, ga[i], 'r', label='Gaussian approxation')
mp.legend()
mp.xlabel('Angular distance [degrees]')
mp.ylabel('Beam (radial cut)')
mp.xlim(-20, 20)
#mp.gcf().savefig('ga-profiles.png', dvi=300)
mp.show()

print 'Total Energy from Interferometry: ', np.sum(syb_ref)
print 'Total Energy from GaussianApprox: ', np.sum(syb_ga)
print (np.sum(syb_ref) - np.sum(syb_ga))/np.sum(syb_ref)*100,'% - Total Diff'
