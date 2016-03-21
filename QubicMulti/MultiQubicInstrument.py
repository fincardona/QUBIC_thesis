from __future__ import division

from qubic import QubicInstrument, QubicScene
from qubic import _flib as flib

from pyoperators import (
    BlockDiagonalOperator, BlockRowOperator, Cartesian2SphericalOperator, 
    DenseBlockDiagonalOperator, DiagonalOperator, HomothetyOperator, 
    IdentityOperator, ReshapeOperator, Rotation3dOperator, 
    Spherical2CartesianOperator, SumOperator, SymmetricBandToeplitzOperator)
from pysimulators import (
    ConvolutionTruncatedExponentialOperator, BeamGaussian, 
    ProjectionOperator)
from pysimulators.interfaces.healpy import (
    Cartesian2HealpixOperator, HealpixConvolutionGaussianOperator)
from pysimulators.sparse import (
    FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix)
from pyoperators.utils import (
    operation_assignment, pool_threading, product, split)
from pyoperators.utils.ufuncs import abs2
from pysimulators.geometry import rotate
from scipy.constants import c, h, k

import numpy as np
import healpy as hp
import numexpr as ne

class MultiQubicInstrument(QubicInstrument):
    
    def __init__(
            self, calibration=None, detector_fknee=0, detector_fslope=1,
            detector_ncorr=10, detector_nep=4.7e-17, detector_ngrids=1,
            detector_tau=0.01, filter_name=150e9, filter_nu=None,
            filter_relative_bandwidth=None, polarizer=True,
            primary_beam=None, secondary_beam=None,
            synthbeam_dtype=np.float32, synthbeam_fraction=0.99,
            synthbeam_kmax=8, synthbeam_peak150_fwhm=np.radians(0.3859879), 
            NPOINTS=1, detector_points=None, NFREQS=1):
        """
        The QubicInstrument class. It represents the instrument setup.

        The exact values of synthbeam_peak150_fwhm are:
        
        mask degrees    mean             sigma             TEST
        
        30      [[[  3.85987922e-01   5.82749218e-06]      energy
                  [  3.79987384e-01   1.30462933e-03]]     residuals
        
        60       [[  3.85987936e-01   5.82815020e-06]      energy
                  [  3.79986166e-01   1.30518810e-03]]]    residuals

        Parameters
        ----------
        NPOINTS : integer in the range: [n**2 for n integer], optional
            Takes into account the spatial extension of the bolometers. It is
            the focal plane number of points inside each detector.
        detector_points : array-like, optional
            Grid of points on the focal plane. They must be conteined inside 
            the detectors. The shape must be: (ndetectors, NPOINTS, 3). If 
            None it will be considered a squared grid of NPOINTS points.
        filter_name : float, optional
            The central wavelength of the band, in Hz: 150e9 or 220e9
        NFREQS : integer, optional
            Takes into account the polychromaticity of the bandwidth. It is 
            the bandwidth number of frequencies. 
        frequencies : array-like, optional
            Grid of frequencies for the bandwidth. If None it will be 
            considered a non-linear grid of NFREQS frequencies.
        relative_bandwidths : array-like, optional
            The relative bandwidths for each frequency of the bandwidth.
        """
        QubicInstrument.__init__(
            self, calibration=calibration, detector_fknee=detector_fknee, 
            detector_fslope=detector_fslope, detector_ncorr=detector_ncorr, 
            detector_nep=detector_nep, detector_ngrids=detector_ngrids,
            detector_tau=detector_tau, polarizer=polarizer, 
            filter_nu=filter_name, filter_relative_bandwidth=0.25,
            primary_beam=primary_beam, secondary_beam=secondary_beam, 
            synthbeam_dtype=synthbeam_dtype, 
            synthbeam_fraction=synthbeam_fraction, 
            synthbeam_kmax=synthbeam_kmax, 
            synthbeam_peak150_fwhm=synthbeam_peak150_fwhm)

        self.filter.name = filter_name
        if filter_nu is None and filter_relative_bandwidth is None:
            self.filter.NFREQS = NFREQS
            self.filter.nu = self.quadratic_grid4freqs()
            self.filter.relative_bandwidth = self.weights() / self.filter.nu
        elif fiter_nu is None and filter_relative_bandwidths is not None:
            raise ValueError("Invalid frequencies or bandwidths")
        elif filter_nu is not None and filter_relative_bandwidths is None:
            raise ValueError("Invalid frequencies or bandwidths")
        else:
            self.filter.NFREQS = len(filter_nu)
            self.filter.nu = filter_nu
            self.filter.relative_bandwidth = filter_relative_bandwidths
        self.filter.bandwidth = (
            self.filter.nu * self.filter.relative_bandwidth)
        if detector_points is None:
            self.detector.NPOINTS = NPOINTS
            side = np.sqrt(self.detector.area)
            self.detector.points = self.shift_grid4pos(
                self.detector.NPOINTS, side, self.detector.center)
        elif len(detector_points.shape) != 3:
                raise ValueError(
                    "The shape must be: (ndetectors, NPOINTS, 3)")
        else:
            self.detector.NPOINTS = detector_points.shape[1]
            self.detector.points = detector_points

    def get_grid4freqs(self):
        on = [130e9, 190e9]
        off = [170e9, 250e9]
        r = [401, 601]
        grid = [np.linspace(cut_on, cut_off, res) for cut_on, cut_off, res in
            zip(on, off, r)]
        if self.filter.name not in (150e9, 220e9):
            raise ValueError("Invalid band '{}'.".format(self.filter.name))
        elif self.filter.name == 150e9:
            return grid[0]
        return grid[1]

    def quadratic_grid4freqs(self):
        grid = self.get_grid4freqs()
        if self.filter.NFREQS==1:
            return np.array([np.mean(grid)])
        elif self.filter.NFREQS==len(grid):
            return grid
        elif self.filter.NFREQS > len(grid):
            raise ValueError("Too many frequencies")
        f = np.arange(self.filter.NFREQS)**2
        return grid[0] + f / f.max() * (grid[-1] - grid[0])
        
    def weights(self):
        grid = self.get_grid4freqs()
        if len(self.filter.nu)==1:
            return np.array([grid[-1] - grid[0]])
        return self._weights(self.filter.nu)

    @staticmethod
    def _weights(freq):
        # weights for integration
        w = np.zeros(len(freq))
        w[0] = 0.5 * (freq[1] - freq[0])
        w[1:-1] = 0.5 * (freq[2:] - freq[:-2])
        w[-1] = 0.5 * (freq[-1] - freq[-2])
        return w

    @staticmethod
    def square_grid4pos(NPOINTS, side):
        n = np.int(np.sqrt(NPOINTS))
        a = np.linspace(0, 1, 2 * n + 1)[range(1, 2 * n + 1, 2)]
        x, y = np.meshgrid(a, a)
        return np.array(zip(x.ravel(), y.ravel())) * side
        
    @staticmethod
    def shift_grid4pos(NPOINTS, side, central_position):
        grid_ = MultiQubicInstrument.square_grid4pos(NPOINTS, side)
        grid = np.full((len(central_position), len(grid_), 2), grid_)
        points = grid + (
            central_position[...,:-1] - np.mean(grid_, axis=0))[:,None,:]
        return np.concatenate((
            points, np.full_like(points[...,0,None], -0.3)), axis=-1)

    @staticmethod
    def _index2coord(nside, index):
        s2c = Spherical2CartesianOperator('zenith,azimuth')
        t, p = hp.pix2ang(nside, index)
        return t, p,s2c(np.concatenate(
            [t[..., None], p[..., None]], axis=-1))
    
    @staticmethod
    def _mask(scene, theta_max):
        """
        Return the sky's indices contained into theta_max

        """
        theta, phi = hp.pix2ang(scene.nside, scene.index)
        ind = np.where(theta <= np.radians(theta_max))[0]
        return hp.ang2pix(scene.nside, theta[ind], phi[ind])

    @staticmethod
    def _masked_beam(scene, beam, theta_max):
        masked_sky = MultiQubicInstrument._mask(scene, theta_max)
        b = np.zeros((len(beam), 12*scene.nside**2))
        b[..., masked_sky] = beam[..., masked_sky]
        return b

    @staticmethod
    def _check_peak(scene, peak, peak_pos, theta_max, dtype): 
        masked_sky = MultiQubicInstrument._mask(scene, theta_max)
        peak_in2_mask = np.zeros(peak_pos.shape, dtype=dtype)
        mask = peak_pos <= np.max(masked_sky)
        peak_in2_mask[mask] = peak[mask]
        return peak_in2_mask 

    @staticmethod
    def _get_aperture_integration_operator(horn):
        """
        Integrate flux density in the telescope aperture.
        Convert signal from W / m^2 / Hz into W / Hz.

        """
        nhorns = np.sum(horn.open)
        return HomothetyOperator(nhorns * np.pi * horn.radius**2)

    def get_aperture_integration_operator(self):
        """
        Integrate flux density in the telescope aperture.
        Convert signal from W / m^2 / Hz into W / Hz.

        """
        
        return self._get_aperture_integration_operator(self.horn)

    @staticmethod
    def _get_convolution_peak_operator(nu, synthbeam, **keywords):
        """
        Return an operator that convolves the Healpix sky by the gaussian
        kernel that, if used in conjonction with the peak sampling operator,
        best approximates the synthetic beam.

        """
        fwhm = synthbeam.peak150.fwhm * (150e9 / nu)
        return HealpixConvolutionGaussianOperator(fwhm=fwhm, **keywords)

    def get_convolution_peak_operator(self, **keywords):
        """
        Return an operator that convolves the Healpix sky by the gaussian
        kernel that, if used in conjonction with the peak sampling operator,
        best approximates the synthetic beam.

        """
        fwhms = self.synthbeam.peak150.fwhm * (150e9 / self.filter.nu)
        fwhm_min = fwhms[-1]
        return HealpixConvolutionGaussianOperator(fwhm=fwhm_min, **keywords)
    
    def get_convolution_transfer_operator(self, **keywords):
        fwhms = self.synthbeam.peak150.fwhm * (150e9 / self.filter.nu)
        fwhm_min = fwhms[-1]
        fwhms_transfer = [np.sqrt(fwhm**2 - fwhm_min**2) for fwhm in fwhms]
        return BlockDiagonalOperator([
            HealpixConvolutionGaussianOperator(fwhm=fwhm, 
            **keywords) for fwhm in fwhms_transfer], new_axisin=0)

    @staticmethod
    def _get_detector_integration_operator(position, area, secondary_beam):
        """
        Integrate flux density in detector solid angles and take into account
        the secondary beam transmission.

        """
        theta = np.arctan2(
            np.sqrt(np.sum(position[..., :2]**2, axis=-1)), position[..., 2])
        phi = np.arctan2(position[..., 1], position[..., 0])
        sr_det = -area / position[..., 2]**2 * np.cos(theta)**3
        sr_beam = secondary_beam.solid_angle
        sec = secondary_beam(theta, phi)
        return DiagonalOperator(
            sr_det / sr_beam * sec, broadcast='rightward')

    def get_detector_integration_operator(self):
        """
        Integrate flux density in detector solid angles and take into account
        the secondary beam transmission.

        """
        return SumOperator(axis=1) * self._get_detector_integration_operator(
            self.detector.points, self.detector.area / self.detector.NPOINTS,
            self.secondary_beam)

    @staticmethod
    def _get_filter_operator(bandwidth):
        """
        Return the filter operator.
        Convert units from W/Hz to W.

        """
        if bandwidth == 0:
            return IdentityOperator()
        return HomothetyOperator(bandwidth)

    def get_filter_operator(self):
        return BlockRowOperator(
            [self._get_filter_operator(bandwidth) for bandwidth in 
             self.filter.bandwidth], new_axisin=0)

    def _get_hwp_operator(self, sampling, scene):
        """
        Return the rotation matrix for the half-wave plate.

        """
        shape = (len(self), self.detector.NPOINTS, len(sampling))
        if scene.kind == 'I':
            return IdentityOperator(shapein=shape)
        if scene.kind == 'QU':
            return Rotation2dOperator(-4 * sampling.angle_hwp,
                                      degrees=True, shapein=shape + (2,))
        return Rotation3dOperator('X', -4 * sampling.angle_hwp,
                                  degrees=True, shapein=shape + (3,))

    def get_hwp_operator(self, sampling, scene):
        return BlockDiagonalOperator([
            self._get_hwp_operator(sampling, scene) for nu in 
            self.filter.nu], new_axisin=0) 
                

    def _get_polarizer_operator(self, sampling, scene):
        """
        Return operator for the polarizer grid.
        When the polarizer is not present a transmission of 1 is assumed
        for the detectors on the first focal plane and of 0 for the other.
        Otherwise, the signal is split onto the focal planes.

        """
        nd = len(self) 
        nP = self.detector.NPOINTS
        nt = len(sampling)
        #grid = self.detector.quadrant // 4
        grid = np.zeros((nd, nP), dtype=np.uint8)

        if scene.kind == 'I':
            if self.optics.polarizer:
                return HomothetyOperator(1 / 2)
            # 1 for the first detector grid and 0 for the second one
            return DiagonalOperator(1 - grid, shapein=(nd, nP, nt),
                                    broadcast='rightward')

        if not self.optics.polarizer:
            raise NotImplementedError(
                'Polarized input is not handled without the polarizer grid.')

        z = np.zeros((nd, nP))
        data = np.array([z + 0.5, 0.5 - grid, z]).T
        data = data.swapaxes(1, 0)[..., None, None, :]
        return ReshapeOperator((nd, nP, nt, 1), (nd, nP, nt)) * \
            DenseBlockDiagonalOperator(data, shapein=(nd, nP, nt, 3))

    def get_polarizer_operator(self, sampling, scene):
        return BlockDiagonalOperator([
            self._get_polarizer_operator(sampling, scene) for nu in 
            self.filter.nu], new_axisin=0)

    @staticmethod
    def _peak_angles_kmax(kmax, horn_spacing, nu, position):
        """
        Return the spherical coordinates (theta, phi) of the beam peaks,
        in radians up to a maximum diffraction order.
        Parameters
        ----------
        kmax : int, optional
            The diffraction order above which the peaks are ignored.
            For instance, a value of kmax=2 will model the synthetic beam by
            (2 * kmax + 1)**2 = 25 peaks and a value of kmax=0 will only 
            sample the central peak.
        horn_spacing : float
            The spacing between horns, in meters.
        nu : float
            The frequency at which the interference peaks are computed.
        position : array of shape (..., 3)
            The focal plane positions for which the angles of the 
            interference peaks are computed.
        """
        lmbda = c / nu    
        position = -position / np.sqrt(  
            np.sum(position**2, axis=-1))[..., None]
        kx, ky = np.mgrid[-kmax:kmax+1, -kmax:kmax+1]
        
        nx = position[..., 0, None] + lmbda * (
            ky.ravel() - kx.ravel()) / horn_spacing / np.sqrt(2)
        ny = position[..., 1, None] - lmbda * (
            kx.ravel() + ky.ravel()) / horn_spacing / np.sqrt(2)
 
        local_dict = {'nx': nx, 'ny': ny} 
        theta = ne.evaluate('arcsin(sqrt(nx**2 + ny**2))',
                            local_dict=local_dict)
        phi = ne.evaluate('arctan2(ny, nx)', local_dict=local_dict)
        return theta, phi

    @staticmethod
    def _peak_angles(scene, nu, position, synthbeam, horn, primary_beam):
        """
        Compute the angles and intensity of the syntheam beam peaks which
        accounts for a specified energy fraction.

        """
        theta, phi = MultiQubicInstrument._peak_angles_kmax(
            synthbeam.kmax, horn.spacing, nu, position)
        val = np.array(primary_beam(theta, phi), dtype=float, copy=False)
        val[~np.isfinite(val)] = 0
        index = _argsort_reverse(val)
        theta = theta[index]
        phi = phi[index]
        val = val[index]
        cumval = np.cumsum(val, axis=-1)
        imaxs = np.argmax(
            cumval >= synthbeam.fraction * cumval[...,-1,None], axis=-1) + 1
        imax = np.max(imaxs)

        # slice initial arrays to discard the non-significant peaks
        theta = theta[..., :imax]
        phi = phi[..., :imax]
        val = val[..., :imax]

        # remove additional per-detector non-significant peaks
        # and remove potential NaN in theta, phi
        for idet, _imax_ in enumerate(imaxs):
            for point, imax_ in enumerate(_imax_): 
                val[idet, point, imax_:] = 0
                theta[idet, point, imax_:] = np.pi / 2 
                                            #XXX 0 fails in polarization.f90.
                                            #src (en2ephi and en2etheta_ephi)
                phi[idet, point, imax_:] = 0
        solid_angle = synthbeam.peak150.solid_angle * (150e9 / nu)**2
        val *= solid_angle / scene.solid_angle * len(horn)
        return theta, phi, val

    @staticmethod
    def _get_projection_operator(
            rotation, scene, nu, position, synthbeam, horn, primary_beam, 
            verbose=True):
        
        if len(position.shape) == 2:
            position = position[None, ...]
        ndetectors = position.shape[0]
        npoints = position.shape[1]
        ntimes = rotation.data.shape[0]
        nside = scene.nside

        thetas, phis, vals = MultiQubicInstrument._peak_angles(
            scene, nu, position, synthbeam, horn, primary_beam)
        
        ncolmax = thetas.shape[-1]
        thetaphi = _pack_vector(thetas, phis)  
                                          # (ndetectors, npoints, ncolmax, 2)
        direction = Spherical2CartesianOperator('zenith,azimuth')(thetaphi)
        e_nf = direction[..., None, :, :]
        if nside > 8192:
            dtype_index = np.dtype(np.int64)
        else:
            dtype_index = np.dtype(np.int32)

        cls = {'I': FSRMatrix,
               'QU': FSRRotation2dMatrix,
               'IQU': FSRRotation3dMatrix}[scene.kind]
        ndims = len(scene.kind)
        nscene = len(scene)
        nscenetot = product(scene.shape[:scene.ndim])
        s = cls((ndetectors * npoints * ntimes * ndims, nscene * ndims), 
                ncolmax=ncolmax, dtype=synthbeam.dtype, 
                dtype_index=dtype_index, verbose=verbose)

        index = s.data.index.reshape((ndetectors, npoints, ntimes, ncolmax))
        c2h = Cartesian2HealpixOperator(nside)
        if nscene != nscenetot:
            table = np.full(nscenetot, -1, dtype_index)
            table[scene.index] = np.arange(len(scene), dtype=dtype_index)

        e_nf = e_nf.reshape(-1, 1, ncolmax, 3)
        index = index.reshape(-1, ntimes, ncolmax)
        def func_thread(i):
            # e_nf[i] shape: (1, ncolmax, 3)
            # e_ni shape: (ntimes, ncolmax, 3)
            e_ni = rotation.T(e_nf[i].swapaxes(0, 1)).swapaxes(0, 1)
            if nscene != nscenetot:
                np.take(table, c2h(e_ni).astype(int), out=index[i])
            else:
                index[i] = c2h(e_ni)

        with pool_threading() as pool:
            pool.map(func_thread, xrange(ndetectors * npoints))
        e_nf = e_nf.reshape(ndetectors, npoints, 1, ncolmax, 3)
        index = index.reshape(ndetectors, npoints, ntimes, ncolmax)
        
        if scene.kind == 'I':
            value = s.data.value.reshape(
                ndetectors, npoints, ntimes, ncolmax)
            value[...] = vals[..., None, :]
            shapeout = (ndetectors, npoints, ntimes)
        else:
            if str(dtype_index) not in ('int32', 'int64') or \
               str(synthbeam.dtype) not in ('float32', 'float64'):
                raise TypeError(
                    'The projection matrix cannot be created with types:'
                    '{0} and {1}.'.format(dtype_index, synthbeam.dtype))
            direction_ = direction.reshape(
                ndetectors * npoints, ncolmax, 3)
            vals_ = vals.reshape(ndetectors * npoints, ncolmax)
            func = 'matrix_rot{0}d_i{1}_r{2}'.format(
                ndims, dtype_index.itemsize, synthbeam.dtype.itemsize)
            getattr(flib.polarization, func)(
                rotation.data.T, direction_.T, s.data.ravel().view(np.int8),
                vals_.T)

            if scene.kind == 'QU':
                shapeout = (ndetectors, npoints, ntimes, 2)
            else:
                shapeout = (ndetectors, npoints, ntimes, 3)
        return ProjectionOperator(s, shapeout=shapeout)
        
    def get_projection_operator(self, sampling, scene, verbose=True):
        """
        Return the peak sampling operator.
        Convert units from W to W/sr.

        Parameters
        ----------
        sampling : QubicSampling
            The pointing information.
        scene : QubicScene
            The observed scene.
        verbose : bool, optional
            If true, display information about the memory allocation.

        """
        horn = getattr(self, 'horn', None)
        primary_beam = getattr(self, 'primary_beam', None)
        rotation = sampling.cartesian_galactic2instrument
        return BlockDiagonalOperator([
            self._get_projection_operator(rotation, scene, nu, 
            self.detector.points, self.synthbeam, horn, primary_beam, 
            verbose=verbose) for nu in self.filter.nu], new_axisin=0)

    @staticmethod
    def _get_response_A(position, area, nu, horn, secondary_beam):
        """
        Phase and transmission from the switches to the focal plane.

        Parameters
        ----------
        position : array-like of shape (..., 3)
            The 3D coordinates where the response is computed [m].
        area : array-like
            The integration area, in m^2.
        nu : float
            The frequency for which the response is computed [Hz].
        horn : PackedArray
            The horn layout.
        secondary_beam : Beam
            The secondary beam.

        Returns
        -------
        out : complex array of shape (#positions, #horns)
            The phase and transmission from the horns to the focal plane.

        """
        uvec = position / np.sqrt(np.sum(position**2, axis=-1))[..., None]
        thetaphi = Cartesian2SphericalOperator('zenith,azimuth')(uvec)
        sr = -area / position[..., 2]**2 * np.cos(thetaphi[..., 0])**3
        tr = np.sqrt(secondary_beam(thetaphi[..., 0], thetaphi[..., 1]) *
                     sr / secondary_beam.solid_angle)[..., None]
        const = 2j * np.pi * nu / c
        product = np.dot(uvec, horn[horn.open].center.T)
        return ne.evaluate('tr * exp(const * product)')

    @staticmethod
    def _get_response_B(
            theta, phi, spectral_irradiance, nu, horn, primary_beam):
        """
        Return the complex electric amplitude and phase [W^(1/2)] from 
        sources of specified spectral irradiance [W/m^2/Hz] going through 
        each horn.

        Parameters
        ----------
        theta : array-like
            The source zenith angle [rad].
        phi : array-like
            The source azimuthal angle [rad].
        spectral_irradiance : array-like
            The source spectral power per unit surface [W/m^2/Hz].
        nu : float
            The frequency for which the response is computed [Hz].
        horn : PackedArray
            The horn layout.
        primary_beam : Beam
            The primary beam.

        Returns
        -------
        out : complex array of shape (#horns, #sources)
            The phase and amplitudes from the sources to the horns.

        """
        shape = np.broadcast(theta, phi, spectral_irradiance).shape
        theta, phi, spectral_irradiance = [np.ravel(_) for _ in theta, phi,
                                           spectral_irradiance]
        uvec = hp.ang2vec(theta, phi)
        source_E = np.sqrt(spectral_irradiance *
                           primary_beam(theta, phi) * np.pi * horn.radius**2)
        const = 2j * np.pi * nu / c
        product = np.dot(horn[horn.open].center, uvec.T)
        out = ne.evaluate('source_E * exp(const * product)')
        return out.reshape((-1,) + shape)

    @staticmethod
    def _get_response(
            theta, phi, spectral_irradiance, position, area, nu, horn, 
            primary_beam, secondary_beam):
        """
        Return the monochromatic complex field [(W/Hz)^(1/2)] related to
        the electric field over a specified area of the focal plane created
        by sources of specified spectral irradiance [W/m^2/Hz]

        Parameters
        ----------
        theta : array-like
            The source zenith angle [rad].
        phi : array-like
            The source azimuthal angle [rad].
        spectral_irradiance : array-like
            The source spectral_irradiance [W/m^2/Hz].
        position : array-like of shape (..., 3)
            The 3D coordinates where the response is computed, in meters.
        area : array-like
            The integration area, in m^2.
        nu : float
            The frequency for which the response is computed [Hz].
        horn : PackedArray
            The horn layout.
        primary_beam : Beam
            The primary beam.
        secondary_beam : Beam
            The secondary beam.

        Returns
        -------
        out : array of shape (#positions, #sources)
            The complex field related to the electric field over a speficied
            area of the focal plane, in units of (W/Hz)^(1/2).

        """
        A = MultiQubicInstrument._get_response_A(
            position, area, nu, horn, secondary_beam)
        B = MultiQubicInstrument._get_response_B(
            theta, phi, spectral_irradiance, nu, horn, primary_beam)
        E = np.dot(A, B.reshape((B.shape[0], -1))).reshape(
            A.shape[:-1] + B.shape[1:])
        return E

    @staticmethod
    def _get_synthbeam_(
            scene, position, area, nu, bandwidth, horn, primary_beam, 
            secondary_beam, spectral_irradiance=1, 
            synthbeam_dtype=np.float32, theta_max=30):
        """
        Return the monochromatic synthetic beam for a specified location
        on the focal plane, multiplied by a given area and bandwidth.

        Parameters
        ----------
        scene : QubicScene
            The scene.
        x : array-like
            The X-coordinate in the focal plane where the response is 
            computed, in meters. If not provided, the detector central 
            positions are assumed.
        y : array-like
            The Y-coordinate in the focal plane where the response is 
            computed, in meters. If not provided, the detector central 
            positions are assumed.
        area : array-like
            The integration area, in m^2.
        nu : float
            The frequency for which the response is computed [Hz].
        bandwidth : float
            The filter bandwidth [Hz].
        horn : PackedArray
            The horn layout.
        primary_beam : Beam
            The primary beam.
        secondary_beam : Beam
            The secondary beam.
        synthbeam_dtype : dtype, optional
            The data type for the synthetic beams (default: float32).
            It is the dtype used to store the values of the pointing matrix.
        theta_max : float, optional
            The maximum zenithal angle above which the synthetic beam is
            assumed to be zero, in degrees.

        """
        MAX_MEMORY_B = 1e9
        theta, phi = hp.pix2ang(scene.nside, scene.index)
        index = np.where(theta <= np.radians(theta_max))[0]
        nhorn = int(np.sum(horn.open))
        npix = len(index)
        nbytes_B = npix * nhorn * 24
        ngroup = np.ceil(nbytes_B / MAX_MEMORY_B)
        out = np.zeros(position.shape[:-1] + (len(scene),),
                       dtype=synthbeam_dtype)
        for s in split(npix, ngroup):
            index_ = index[s]
            sb = MultiQubicInstrument._get_response(
                theta[index_], phi[index_], spectral_irradiance, position, 
                area, nu, horn, primary_beam, secondary_beam)
            out[..., index_] = abs2(sb, dtype=synthbeam_dtype)
        return out * bandwidth * deriv_and_const(nu, scene.nside)

    @staticmethod
    def _get_synthbeam(
            scene, position, area, freqs, bandwidths, horn, primary_beam, 
            secondary_beam, spectral_irradiance=1, 
            synthbeam_dtype=np.float32, theta_max=30):
        sb = np.zeros((position.shape[0], 12 * scene.nside**2))
        for nu, bandwidth in zip(freqs, bandwidths):
            sb += MultiQubicInstrument._get_synthbeam_(
                scene, position, area, nu, bandwidth, horn, primary_beam, 
                secondary_beam, synthbeam_dtype=synthbeam_dtype, 
                theta_max=theta_max)
        return np.sum(sb, axis=0)

    def get_synthbeam(self, scene, idet=None, theta_max=30, i=None):
        """
        Return the detector synthetic beams, computed from the superposition
        of the electromagnetic fields.

        The synthetic beam B_d = (B_d,i) of a given detector d is such that
        the power I_d in [W] collected by this detector observing a sky 
        S=(S_i) in [W/m^2/Hz] is:
            I_d = (S | B_d) = sum_i S_i * B_d,i.

        Example
        -------
        >>> scene = QubicScene(1024)
        >>> inst = QubicInstrument()
        >>> sb = inst.get_synthbeam(scene, 0)

        

        Parameters
        ----------
        scene : QubicScene
            The scene.
        idet : int, optional
            The detector number. By default, the synthetic beam is computed 
            for all detectors.
        theta_max : float, optional
            The maximum zenithal angle above which the synthetic beam is
            assumed to be zero, in degrees.

        """
        if idet is not None:
            return self[idet].get_synthbeam(
                scene, theta_max=theta_max, i=idet)
        return self._get_synthbeam(
            scene, self.detector.points[i], 
            self.detector.area / self.detector.NPOINTS, self.filter.nu, 
            self.filter.bandwidth, self.horn, self.primary_beam, 
            self.secondary_beam, synthbeam_dtype=self.synthbeam.dtype, 
            theta_max=theta_max) 
   
    @staticmethod
    def _direct_convolution_(
            scene, position, area, nu, bandwidth, horn, primary_beam, 
            secondary_beam, synthbeam, theta_max=30):
        
        rotation = DenseBlockDiagonalOperator(np.eye(3)[None, ...])
        aperture = MultiQubicInstrument._get_aperture_integration_operator(
            horn)
        scene_nopol = QubicScene(scene.nside, kind='I')
        projection = MultiQubicInstrument._get_projection_operator(
            rotation, scene_nopol, nu, position, synthbeam, horn, 
            primary_beam, verbose=False)
        masked_sky = MultiQubicInstrument._mask(scene, theta_max)
        peak_pos = projection.matrix.data.index
        pos = MultiQubicInstrument._check_peak(
            scene, peak_pos, peak_pos, theta_max, dtype=np.int32)
        value = MultiQubicInstrument._check_peak(
            scene, projection.matrix.data.value, peak_pos, theta_max, 
            dtype=np.float64)
        integ = MultiQubicInstrument._get_detector_integration_operator(
            position, area, secondary_beam)
        b = BeamGaussian(synthbeam.peak150.fwhm / nu * 150e9)
        theta0, phi0, uvec0 = MultiQubicInstrument._index2coord(
            scene.nside, pos)
        theta, phi, uvec = MultiQubicInstrument._index2coord(
            scene.nside, masked_sky)
        dot = np.dot(uvec0, uvec.T)
        dot[dot>1] = 1
        dtheta = np.arccos(dot) # (#npeaks, #pixels_inside_mask)
        maps = b(dtheta, 0)
        maps /= np.sum(maps, axis=-1)[..., None]
        Map = np.sum((maps * value[..., None]), axis=1)
        Map = integ(aperture(Map)) 
        Map *= bandwidth * deriv_and_const(nu, scene.nside)
        return MultiQubicInstrument._masked_beam(
            scene, Map, theta_max)

    @staticmethod
    def _direct_convolution(
            scene, position, area, freqs, bandwidths, horn, primary_beam, 
            secondary_beam, synthbeam, theta_max=30):
        sb = np.zeros((position.shape[0], 12 * scene.nside**2))
        for nu, bandwidth in zip(freqs, bandwidths):
            sb += MultiQubicInstrument._direct_convolution_(
                scene, position, area, nu, bandwidth, horn, primary_beam, 
                secondary_beam, synthbeam, theta_max)
        return np.sum(sb, axis=0)

    def direct_convolution(self, scene, idet=None, theta_max=30, i=None):
        if idet is not None:
            return self[idet].direct_convolution(
                scene, theta_max=theta_max, i=idet)
        return self._direct_convolution(
            scene, self.detector.points[i], 
            self.detector.area / self.detector.NPOINTS, self.filter.nu,  
            self.filter.bandwidth, self.horn, self.primary_beam, 
            self.secondary_beam, self.synthbeam,  theta_max)
    
class MyBeam(object):
    def __init__(
            self, func, fwhm1, fwhm2, x0_1, x0_2, weight1, weight2, nside, 
            backward=False):
        self.func = func
        self.fwhm1 = fwhm1
        self.fwhm2 = fwhm2
        self.x0_1 = x0_1
        self.x0_2 = x0_2
        self.weight1 = weight1
        self.weight2 = weight2
        self.nside = nside
        self.backward = backward
        npix = 12*nside**2
        pix = np.arange(12*nside**2)
        theta, phi = hp.pix2ang(nside, pix)
        self.norm = self.func(
            0, 0, self.fwhm1, self.fwhm2, self.x0_1, 
            self.x0_2, self.weight1, self.weight2)
        if self.backward:
            theta = np.pi - theta
        self.solid_angle = np.sum(
            func(theta, phi, self.fwhm1, self.fwhm2, self.x0_1, self.x0_2, 
            self.weight1, self.weight2) * 4 * np.pi / npix) / self.norm
    def __call__(self, theta, phi):
        if self.backward:
            theta = np.pi - theta
        return self.func(
            theta, phi, self.fwhm1, self.fwhm2, self.x0_1, self.x0_2, 
            self.weight1, self.weight2) / self.norm

def _argsort_reverse(a, axis=-1):
    i = list(np.ogrid[[slice(x) for x in a.shape]])
    i[axis] = a.argsort(axis)[..., ::-1]
    return i

def _pack_vector(*args):
    shape = np.broadcast(*args).shape
    out = np.empty(shape + (len(args),))
    for i, arg in enumerate(args):
        out[..., i] = arg
    return out

def deriv_and_const(nu, nside):
    # Power contribution for each pixel of the sky
    T = QubicScene().T_cmb
    return (8 * np.pi / (12 * nside**2) * 1e-6 * h**2 * nu**4 / 
            (k * c**2 * T**2) * np.exp(h * nu / (k * T)) / 
            (np.expm1(h * nu / (k * T)))**2)
