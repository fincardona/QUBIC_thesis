from __future__ import division

import matplotlib.pyplot as mp
import numpy as np
from pyoperators import MaskOperator
from pysimulators import create_fitsheader, SceneGrid
from qubic import QubicInstrument
from MyQubicInstrument import MyQubicInstrument, MyBeam 

def gaussian_beam(theta, phi, fwhm, x0=0):
    sigma = np.radians(fwhm / (2 * np.sqrt(2 * np.log(2))))
    return np.exp(- (theta-x0)**2 / (2 * sigma**2))

def alternative_beam(theta, phi):
    fwhm1 = 13
    fwhm2 = 7
    x0_1 = 0
    x0_2 = 0.2 
    return gaussian_beam(theta, phi, fwhm1, x0_1)*6/8 + gaussian_beam(
        theta, phi, fwhm2, x0_2)*2/8
    
theta = np.linspace(0, np.pi, 1000)
# mp.plot(theta, alternative_beam(theta, 0))
# mp.plot(theta, gaussian_beam(theta, 0, 13))

NU = 220e9                   # [Hz]

SOURCE_POWER = 1             # [W]
SOURCE_THETA = np.radians(3) # [rad]
SOURCE_PHI = np.radians(45)  # [rad]

NPOINT_FOCAL_PLANE = 512**2  # number of detector plane sampling points

NSIDE = 512

#primary_beam = MyBeam(alternative_beam, fwhm1, fwhm2, x0_1, x0_2, weight1, weight2, NSIDE)
#secondary_beam = MyBeam(alternative_beam, fwhm1, fwhm2, x0_1, x0_2, weight1, weight2, NSIDE, backward=True)

#qubic = MyQubicInstrument(
#    primary_beam=primary_beam, secondary_beam=secondary_beam, 
#    detector_ngrids=1, filter_nu=NU)

qubic = QubicInstrument(detector_ngrids=1, filter_nu=NU)

qubic.horn.open[:] = False
#qubic.horn.open[10] = True
#qubic.horn.open[14] = True
qubic.horn.open[78] = True
qubic.horn.open[82] = True
qubic.horn.plot(facecolor_closed='white', facecolor_open='red')

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
header = create_fitsheader((nfp_x, nfp_x), cdelt=fp_spacing, crval=(0, 0),
                           ctype=['X---CAR', 'Y---CAR'], cunit=['m', 'm'])
focal_plane = SceneGrid.fromfits(header)
integ = MaskOperator(qubic.detector.all.removed) * \
        focal_plane.get_integration_operator(
            focal_plane.topixel(qubic.detector.all.vertex[..., :2]))


###############
# COMPUTATIONS
###############
E = qubic._get_response(SOURCE_THETA, SOURCE_PHI, SOURCE_POWER,
                        fp, fp_spacing**2, qubic.filter.nu, qubic.horn,
                        qubic.primary_beam, qubic.secondary_beam)
I = np.abs(E)**2
D = integ(I)


##########
# DISPLAY
##########
mp.figure()
mp.imshow(np.log10(I), interpolation='nearest', origin='lower')
mp.autoscale(False)
qubic.detector.plot(transform=focal_plane.topixel, linewidth=0.3)
mp.figure()
mp.imshow(np.log(D), interpolation='nearest')
mp.gca().format_coord = lambda x, y: 'x={} y={} z={}'.format(x, y, D[x, y])
mp.show()
print('Given {} horns, we get {} W in the detector plane and {} W in the detec'
      'tors.'.format(int(np.sum(qubic.horn.open)), np.sum(I), np.sum(D)))
