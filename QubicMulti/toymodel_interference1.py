from __future__ import division

from matplotlib import gridspec 
from MultiQubicInstrument import MultiQubicInstrument, MyBeam 
from pyoperators import MaskOperator
from pysimulators import create_fitsheader, SceneGrid
from qubic import QubicInstrument

import matplotlib.pyplot as mp
import numpy as np

def plott(I, save=False):
    limits = np.round(np.arange(-0.052, 0.052, 0.017333*2), 3)
    limits[limits==0] = 0
    #mp.figure()
    mp.imshow(10*np.log10(I/I.max()), interpolation='nearest', 
              origin='lower', vmin=-70)
    colb = mp.colorbar()
    colb.set_label('dB', fontsize='x-large')
    mp.autoscale(False)
    ylim = 680
    mp.xticks(np.arange(0, ylim, ylim / len(limits)), limits)
    mp.yticks(np.arange(0, ylim, ylim / len(limits)), limits)
    qubic.detector.plot(transform=focal_plane.topixel, 
                        linewidth=0.3)
    if save:
        mp.gcf().savefig('toymodel.png', dvi=300)
    mp.show()

NU = 150e9                   # [Hz]

SOURCE_POWER = 1             # [W]
SOURCE_THETA = np.radians(0) # [rad]
SOURCE_PHI = np.radians(0)  # [rad]

NPOINT_FOCAL_PLANE = 512**2  # number of detector plane sampling points

NSIDE = 512

qubic = MultiQubicInstrument(detector_ngrids=1, filter_name=NU)
qubic.horn.open[:] = False
qubic.horn.open[10] = True
qubic.horn.open[14] = True
qubic.horn.open[78] = True
qubic.horn.open[82] = True

FOCAL_PLANE_LIMITS = (np.nanmin(qubic.detector.vertex[..., 0]),
                      np.nanmax(qubic.detector.vertex[..., 0]))  # [m]
# to check energy conservation (unrealistic detector plane):
#FOCAL_PLANE_LIMITS = (-4, 4) # [m]
#FOCAL_PLANE_LIMITS = (-0.2, 0.2) # [m]

#################
# FOCAL PLANE
#################
nfp_x = int(np.sqrt(NPOINT_FOCAL_PLANE))
a = np.r_[FOCAL_PLANE_LIMITS[0]:FOCAL_PLANE_LIMITS[1]:nfp_x*1j]
fp_x, fp_y = np.meshgrid(a, a)
fp = np.dstack([fp_x, fp_y, np.full_like(fp_x, -qubic.optics.focal_length)])
fp_spacing = (FOCAL_PLANE_LIMITS[1] - FOCAL_PLANE_LIMITS[0]) / nfp_x
############
# DETECTORS
############
header = create_fitsheader((nfp_x, nfp_x), cdelt=fp_spacing, 
                           crval=(0, 0), 
                           ctype=['X---CAR', 'Y---CAR'], 
                           cunit=['m', 'm'])
focal_plane = SceneGrid.fromfits(header)
integ = MaskOperator(qubic.detector.all.removed) * \
        focal_plane.get_integration_operator(
            focal_plane.topixel(qubic.detector.all.vertex[..., :2]))
###############
# COMPUTATIONS
###############
E = qubic._get_response(SOURCE_THETA, SOURCE_PHI, SOURCE_POWER,
                        fp, fp_spacing**2, qubic.filter.nu, 
                        qubic.horn, qubic.primary_beam, 
                        qubic.secondary_beam)
I = np.abs(E)**2
D = integ(I)
print('Given {} horns, we get {} W in the detector plane and {} W in the detec'
      'tors.'.format(int(np.sum(qubic.horn.open)), np.sum(I), np.sum(D)))
##########
# DISPLAY
##########

gs = gridspec.GridSpec(2, 2, width_ratios=[3, 4])

mp.figure()
mp.subplot(gs[0])
qubic.horn.plot(facecolor_closed='white', facecolor_open='red')
mp.xticks(np.arange(-0.2, 0.28, 0.08), np.arange(-0.2, 0.28, 0.08))
mp.yticks(np.arange(-0.2, 0.28, 0.08), np.arange(-0.2, 0.28, 0.08))
mp.subplot(gs[1])
plott(I, save=False)

qubic.horn.open[:] = False
qubic.horn.open[10] = True
qubic.horn.open[14] = False
qubic.horn.open[78] = True
qubic.horn.open[82] = False
mp.subplot(gs[2])
qubic.horn.plot(facecolor_closed='white', facecolor_open='red')
mp.xticks(np.arange(-0.2, 0.28, 0.08), np.arange(-0.2, 0.28, 0.08))
mp.yticks(np.arange(-0.2, 0.28, 0.08), np.arange(-0.2, 0.28, 0.08))

E = qubic._get_response(SOURCE_THETA, SOURCE_PHI, SOURCE_POWER,
                        fp, fp_spacing**2, qubic.filter.nu, 
                        qubic.horn, qubic.primary_beam, 
                        qubic.secondary_beam)
I = np.abs(E)**2
mp.subplot(gs[3])
plott(I, save=False)
