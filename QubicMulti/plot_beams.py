from __future__ import division

from MultiQubicInstrument import MultiQubicInstrument
from qubic import QubicScene

import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

nu_cent = 150e9  # the central frequency of the bandwidth
nside = 1024
scene = QubicScene(nside)
idet = 231  # one of 4 central detectors
theta_max = 30
syb_f = 0.99

NFREQS = 1  # monochromatic
NPOINTS = 1 # no size of detectors

q = MultiQubicInstrument(
    filter_name=nu_cent, synthbeam_fraction=syb_f, NFREQS=NFREQS, 
    NPOINTS=NPOINTS)

# Synthesized Beams & Gaussian Approximation
sb = q.get_synthbeam(scene, idet, theta_max) # the interferometric beam 
sb_ga = q.direct_convolution(scene, idet, theta_max)  # the approximated one

# decibel (dB) conversion
sb_dB = 10 * np.log10(sb/sb.max())
sb_ga_dB = 10 * np.log10(sb_ga/sb_ga.max())

# Display logarithmic beams
hp.gnomview(sb_dB, rot=[0, 90], reso=1, xsize=2000, unit='dB') # play with reso and xsize
hp.gnomview(sb_ga_dB, rot=[0, 90], reso=1, xsize=2000, unit='dB') # horrible 

# Display beams
sb_ = hp.gnomview(
    sb, rot=[0, 90], reso=5, xsize=600, min=0, max=np.max(sb), 
    return_projected_map=True, title='Gaussian approximation',
    margins=4 * [0.01])

ga = hp.gnomview(
    sb_ga, rot=[0, 90], reso=5, xsize=600, min=0, max=np.max(sb_ga), 
    return_projected_map=True, title='Gaussian approximation',
    margins=4 * [0.01])

# Display profiles
mp.figure()
i, j = np.unravel_index(np.argmax(ga), ga.shape)
x = np.arange(600) * 5 / 60
x -= x[j]
mp.plot(x, sb_[i], 'g', label='Interferometry')
mp.plot(x, ga[i], 'r', label='Gaussian approxation')
mp.legend()
mp.xlabel('Angular distance [degrees]')
mp.ylabel('Beam (radial cut)')
mp.xlim(-20, 20)
#mp.ylim(-0.1, 1.1)
#mp.gcf().savefig('ga-profiles.png', dvi=300)
mp.show()
