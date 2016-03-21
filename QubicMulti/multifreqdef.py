from __future__ import division

from argparse import ArgumentParser
from MultiQubicInstrument import MultiQubicInstrument
from MultiQubicScene import MultiQubicScene
from MultiQubicAcquisition import (
    MultiPlanckAcquisition, MultiQubicAcquisition)
from pyoperators import MPI, pcg
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator)
from qubic import (
    create_random_pointings, QubicAcquisition,
    QubicPlanckAcquisition, QubicInstrument, QubicScene)
from qubic.data import PATH
from qubic.io import read_map, write_map
from scipy.constants import c, h, k
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import gc

def reconstruction(
        racenter, deccenter, NPOINTS, NFREQS, filter_name, scene, 
        maxiter, tol, rank, n, start):
    sampling = create_random_pointings([racenter, deccenter], 1000, 10)
    detector_nep = 4.7e-17 * np.sqrt(
        len(sampling) * sampling.period / (365 * 86400))

    #x0 = read_map(PATH + 'syn256_pol.fits')
    x0 = np.zeros((12*nside**2, 3))
    q = MultiQubicInstrument(
        NPOINTS=NPOINTS, NFREQS=NFREQS, filter_name=filter_name, 
        detector_nep=detector_nep)
    
    C_nf = q.get_convolution_peak_operator()
    conv_sky_ = C_nf(x0)
    
    fwhm_t = np.sqrt(q.synthbeam.peak150.fwhm**2 - C_nf.fwhm**2)
    C_transf = HealpixConvolutionGaussianOperator(fwhm=fwhm_t)
    
    acq = MultiQubicAcquisition(q, sampling, scene=scene)
    H = acq.get_operator()
    coverage = acq.get_coverage(H)
    
    acq_planck = MultiPlanckAcquisition(
        np.int(filter_name/1e9), scene, true_sky=conv_sky_) 
    acq_fusion = QubicPlanckAcquisition(acq, acq_planck)
    
    H = acq_fusion.get_operator()
    y = acq_fusion.get_observation()
    invntt = acq_fusion.get_invntt_operator()
    
    A = H.T * invntt * H
    b = H.T * invntt * y
    
    solution_fusion = pcg(A, b, disp=True, maxiter=maxiter, tol=tol)
    x_rec_fusion_ = solution_fusion['x']
    x_rec_fusion = C_transf(x_rec_fusion_)

    #H = acq.get_operator()
    #COV = acq.get_coverage(H)
    #y = acq.get_observation(conv_sky_)
    #invntt = acq.get_invntt_operator()
    
    # A = H.T * invntt * H
    # b = H.T * invntt * y
    
    # solution_qubic = pcg(A, b, disp=True, maxiter=maxiter, tol=tol)
    # x_rec_qubic_ = solution_qubic['x']
    # x_rec_qubic = C_transf(x_rec_qubic_)

    # conv_sky = C_transf(conv_sky_)
    
    path = '/home/fincardona/Qubic/map_making/maps/montecarlo'
    if rank==0:
        hp.write_map('%s/fusion_n{}_N{}_s{}_mi{}_niter{}.fits'.format(
            NFREQS, NPOINTS, len(sampling), maxiter, n+start) % path, 
            x_rec_fusion.T)
        #hp.write_map('maps_test/qubic_n{}_N{}_s{}_mi{}.fits'.format(
        #    NFREQS, NPOINTS, len(sampling), maxiter), x_rec_qubic.T)
    gc.collect()
    return coverage, sampling, path


rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

parser = ArgumentParser()
parser.add_argument('NFREQS', type=int, default=1)
parser.add_argument('NPOINTS', type=int, default=1)
parser.add_argument('start', type=int, default=1)
parser.add_argument('niter', type=int, default=1)
args = parser.parse_args()

filter_name = 150e9
NFREQS = args.NFREQS
NPOINTS = args.NPOINTS

nside = 256
scene = MultiQubicScene(nside)
T_cmb = scene.T_cmb

maxiter = 1000
tol = 5e-6

#np.random.seed(0)
racenter = 0.0      # deg
deccenter = -57.0   # deg

for n in range(args.niter):
    coverage, sampling, path = reconstruction(
        racenter, deccenter, NPOINTS, NFREQS, filter_name, scene, 
        maxiter, tol, rank, n, args.start)
    
hp.write_map('%s/qcoverage_n{}_N{}_s{}_mi{}.fits'.format(
    NFREQS, NPOINTS, len(sampling), maxiter) % path, coverage.T)
