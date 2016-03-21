# coding: utf-8
from __future__ import division

import astropy.io.fits as pyfits
import healpy as hp
import numpy as np
import os
import time
import yaml
from glob import glob
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator,
    CompositionOperator, DiagonalOperator, I, IdentityOperator,
    MPIDistributionIdentityOperator, MPI, proxy_group, ReshapeOperator,
    rule_manager)
from pyoperators.utils import ifirst
from pyoperators.utils.mpi import as_mpi
from pysimulators import Acquisition, FitsArray, ProjectionOperator
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator)
from qubic import PlanckAcquisition, QubicAcquisition, QubicPlanckAcquisition
from qubic.calibration import QubicCalibration
from qubic.data import PATH
from qubic.instrument import QubicInstrument, SimpleInstrument
from qubic.scene import QubicScene

__all__ = ['MultiQubicAcquisition']

class MultiQubicAcquisition(QubicAcquisition):
    """
    The MultiQubicAcquisition class, which combines the instrument, 
    sampling and scene models.

    """
    def __init__(self, instrument, sampling, scene=None, block=None,
                 calibration=None, detector_nep=4.7e-17, detector_fknee=0,
                 detector_fslope=1, detector_ncorr=10, detector_ngrids=1,
                 detector_tau=0.01, filter_relative_bandwidth=0.25,
                 polarizer=True,  primary_beam=None, secondary_beam=None,
                 synthbeam_dtype=np.float32, synthbeam_fraction=0.99,
                 absolute=False, kind='IQU', nside=256, max_nbytes=None,
                 nprocs_instrument=None, nprocs_sampling=None, comm=None):
        """
        acq = MultiQubicAcquisition(band, sampling,
                               [scene=|absolute=, kind=, nside=],
                               nprocs_instrument=, nprocs_sampling=, comm=)
        acq = MultiQubicAcquisition(instrument, sampling,
                               [scene=|absolute=, kind=, nside=],
                               nprocs_instrument=, nprocs_sampling=, comm=)

        Parameters
        ----------
        band : int
            The module nominal frequency, in GHz.
        scene : QubicScene, optional
            The discretized observed scene (the sky).
        block : tuple of slices, optional
            Partition of the samplings.
        absolute : boolean, optional
            If true, the scene pixel values include the CMB background and 
            the fluctuations in units of Kelvin, otherwise it only represent 
            the fluctuations, in microKelvin.
        kind : 'I', 'QU' or 'IQU', optional
            The sky kind: 'I' for intensity-only, 'QU' for Q and U maps,
            and 'IQU' for intensity plus QU maps.
        nside : int, optional
            The Healpix scene's nside.
        instrument : QubicInstrument, optional
            The QubicInstrument instance.
        calibration : QubicCalibration, optional
            The calibration tree.
        detector_fknee : array-like, optional
            The detector 1/f knee frequency in Hertz.
        detector_fslope : array-like, optional
            The detector 1/f slope index.
        detector_ncorr : int, optional
            The detector 1/f correlation length.
        detector_nep : array-like, optional
            The detector NEP [W/sqrt(Hz)].
        detector_ngrids : 1 or 2, optional
            Number of detector grids.
        detector_tau : array-like, optional
            The detector time constants in seconds.
        filter_relative_bandwidth : float, optional
            The filter relative bandwidth delta_nu/nu.
        polarizer : boolean, optional
            If true, the polarizer grid is present in the optics setup.
        primary_beam : function f(theta [rad], phi [rad]), optional
            The primary beam transmission function.
        secondary_beam : function f(theta [rad], phi [rad]), optional
            The secondary beam transmission function.
        synthbeam_dtype : dtype, optional
            The data type for the synthetic beams (default: float32).
            It is the dtype used to store the values of the pointing matrix.
        synthbeam_fraction: float, optional
            The fraction of significant peaks retained for the computation
            of the synthetic beam.
        max_nbytes : int or None, optional
            Maximum number of bytes to be allocated for the acquisition's
            operator.
        nprocs_instrument : int, optional
            For a given sampling slice, number of procs dedicated to
            the instrument.
        nprocs_sampling : int, optional
            For a given detector slice, number of procs dedicated to
            the sampling.
        comm : mpi4py.MPI.Comm, optional
            The acquisition's MPI communicator. Note that it is transformed
            into a 2d cartesian communicator before being stored as the 
            'comm' attribute. The following relationship must hold:
                comm.size = nprocs_instrument * nprocs_sampling

        """
        QubicAcquisition.__init__(
            self, instrument=instrument, sampling=sampling, scene=scene, 
            block=block, calibration=calibration, detector_nep=detector_nep, 
            detector_fknee=detector_fknee, detector_fslope=detector_fslope, 
            detector_ncorr=detector_ncorr, detector_ngrids=detector_ngrids,
            detector_tau=detector_tau, 
            filter_relative_bandwidth=filter_relative_bandwidth, 
            polarizer=polarizer, primary_beam=primary_beam, 
            secondary_beam=secondary_beam, synthbeam_dtype=synthbeam_dtype, 
            synthbeam_fraction=synthbeam_fraction, absolute=absolute, 
            kind=kind, nside=nside, max_nbytes=max_nbytes, 
            nprocs_instrument=nprocs_instrument, 
            nprocs_sampling=nprocs_sampling, comm=comm)

    def get_coverage(self, H, top=100):
        coverage =  H.T(
            np.ones((len(self.instrument), len(self.sampling))))[:, 0]
        cov = np.zeros(len(coverage))
        mask = coverage > coverage.max() / top
        cov[mask] = coverage[mask]
        return cov / cov.max()

    def get_operator(self):
        """
        Return the operator of the acquisition. Note that the operator is 
        only linear if the scene temperature is differential (absolute=False)
        """
        distribution = self.get_distribution_operator()
        temp = self.get_unit_conversion_operator()
        aperture = self.get_aperture_integration_operator()
        projection = self.get_projection_operator()
        hwp = self.get_hwp_operator()
        polarizer = self.get_polarizer_operator()
        filter = self.get_filter_operator()
        integ = self.get_detector_integration_operator()
        response = self.get_detector_response_operator()
        convol = self.instrument.get_convolution_transfer_operator()
     
        with rule_manager(inplace=True):
            H = CompositionOperator([
                response, integ, filter, polarizer, hwp * projection, 
                aperture, convol, temp, distribution])
        if self.scene == 'QU':
            H = self.get_subtract_grid_operator()(H)
        return H

