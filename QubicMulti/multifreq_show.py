from __future__ import division
from pyoperators import (
    AdditionOperator, HomothetyOperator, MaskOperator, operation_assignment, 
    pcg, ReshapeOperator, SumOperator)
from pysimulators import profile
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator)
from qubic import (
    create_random_pointings, equ2gal, PlanckAcquisition, QubicAcquisition,
    QubicPlanckAcquisition, QubicInstrument, QubicScene)
from MultiQubicInstrument import MultiQubicInstrument
from MultiQubicScene import MultiQubicScene
from MultiQubicAcquisition import MultiQubicAcquisition
from qubic.data import PATH
from qubic.io import read_map, write_map
from scipy.constants import c, h, k
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np

filter_name = 150e9
NFREQS = 3
NPOINTS = 4

nside = 256
scene = MultiQubicScene(nside)
T_cmb = scene.T_cmb

#np.random.seed(0)
racenter = 0.0      # deg
deccenter = -57.0   # deg
sampling = create_random_pointings([racenter, deccenter], 1000, 10)
detector_nep = 4.7e-17 * np.sqrt(
    len(sampling) * sampling.period / (365 * 86400))

maxiter = 1000

x0 = read_map(PATH + 'syn256_pol.fits')

q = MultiQubicInstrument(
    NPOINTS=NPOINTS, NFREQS=NFREQS, filter_name=filter_name, 
    detector_nep=detector_nep)

C_nf = q.get_convolution_peak_operator()
conv_sky_ = C_nf(x0)

fwhm_t = np.sqrt(q.synthbeam.peak150.fwhm**2 - C_nf.fwhm**2)
C_transf = HealpixConvolutionGaussianOperator(fwhm=fwhm_t)

acq = MultiQubicAcquisition(q, sampling, scene=scene)
acq_planck = PlanckAcquisition(np.int(filter_name/1e9), scene, 
                               true_sky=conv_sky_) 
H = acq.get_operator()
COV = acq.get_coverage(H)

conv_sky = C_transf(conv_sky_)

x_rec_fusion = read_map('maps_test/fusion_n{}_N{}_s{}_mi{}.fits'.format(
                        NFREQS, NPOINTS, len(sampling), maxiter))

x_rec_qubic = read_map('maps_test/qubic_n{}_N{}_s{}_mi{}.fits'.format(NFREQS,
                       NPOINTS, len(sampling), maxiter))

# some display
def display(input, msg, iplot=1):
    out = []
    for i, (kind, lim) in enumerate(zip('IQU', [50, 5, 5])):
        map = input[..., i]
        out += [hp.gnomview(
            map, rot=center, reso=5, xsize=800, min=-lim, max=lim, 
            title=msg + ' ' + kind, sub=(3, 3, iplot + i), 
            return_projected_map=True)]
    return out

center = equ2gal(racenter, deccenter)

mp.figure(1)
mp.clf()
x_rec_qubic[COV == 0] = np.nan
display(conv_sky, 'Original map', iplot=1)
display(x_rec_qubic, 'Reconstructed map', iplot=4)
res_qubic = display(x_rec_qubic - conv_sky, 'Difference map', iplot=7)

mp.figure(2)
mp.clf()
display(conv_sky, 'Original map', iplot=1)
display(x_rec_fusion, 'Reconstructed map', iplot=4)
res_fusion = display(x_rec_fusion - conv_sky, 'Difference map', iplot=7)

mp.figure(3)
for res, color in zip((res_qubic, res_fusion), ('blue', 'green')):
    for i, kind in enumerate('IQU'):
        axis = mp.subplot(3, 1, i+1)
        x, y = profile(res[i]**2)
        x *= 5 / 60
        y = np.sqrt(y)
        y *= np.degrees(np.sqrt(4 * np.pi / acq.scene.npixel))
        mp.plot(x, y, color=color)
        mp.title(kind)
        mp.ylim(0, 1.8)
        mp.ylabel('Sensitivity [$\mu$K deg]')
mp.xlabel('Angular distance [degrees]')

# BICEP-2 / Planck
# sigmas = 1.2 * np.array([1 / np.sqrt(2), 1, 1])
sigmas = np.std(acq_planck.sigma, 0)
for i, sigma in enumerate(sigmas):
    axis = mp.subplot(3, 1, i+1)
    mp.axhline(sigma, color='red')

hp.mollzoom(x_rec_fusion[:, 0] - conv_sky[:, 0])

mp.show()
