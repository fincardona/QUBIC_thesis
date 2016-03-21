#coding: utf8
from __future__ import division

import gc
import healpy as hp
import matplotlib.pyplot as mp
import numpy as np
import scipy as sp
import time

from pysimulators.interfaces.healpy import SceneHealpixCMB
from pysimulators.geometry import rotate
from qubic import QubicInstrument, QubicScene
from qubic.utils import progress_bar
from MyQubicInstrument import MyQubicInstrument

def sybs(fwhm150, scene, nu, idet, theta_max, syb_f):
    q = MyQubicInstrument(
        filter_nu=nu, filter_relative_bandwidth=1/nu,
        synthbeam_dtype=float, synthbeam_fraction=syb_f, 
        synthbeam_peak150_fwhm=np.radians(fwhm150))
    syb_ref = q.get_synthbeam(scene, idet, theta_max=theta_max)
    syb_ga = q.direct_convolution(scene, idet, theta_max=theta_max)
    return syb_ref, syb_ga

def deltaEoverE(fwhm150, scene, nu, idet, theta_max, syb_f):
    syb_ref, syb_ga = sybs(fwhm150, scene, nu, idet, theta_max, syb_f)
    return (np.sum(syb_ref) - np.sum(syb_ga)) / np.sum(syb_ref)

def residuals(fwhm150, scene, nu, idet, theta_max, syb_f):
    syb_ref, syb_ga = sybs(fwhm150, scene, nu, idet, theta_max, syb_f)
    syb_ref /= np.sum(syb_ref)
    syb_ga /= np.sum(syb_ga)
    return np.sum((syb_ref - syb_ga)**2)

nside = 1024   
scene = SceneHealpixCMB(nside, kind='I') 

nu = 150e9     #fixed
syb_f = 1      #fixed

det = np.arange(0, 992)

t_max = [30, 60] 

bounds = [0.37, 0.39]

fwhm150 = np.zeros((len(t_max), 2)) 
bar = progress_bar(len(det))
for idet in det:
    for j, theta_max in enumerate(t_max):
        fwhm150_energy = sp.optimize.brentq(
            deltaEoverE, bounds[0], bounds[1], args=(
            scene, nu, idet, theta_max, syb_f))
        fwhm150_residuals = sp.optimize.minimize_scalar(
            residuals, args=(scene, nu, idet, theta_max, syb_f), 
            bounds=(bounds[0], bounds[1]), method='bounded')
        fwhm150[j] = [fwhm150_energy, fwhm150_residuals.x]
        np.save('fwhm150s/fwhm150_nside{}_det{}_sybf{}_thetamax{}'.format(
            nside, idet, syb_f, t_max), fwhm150)
        gc.collect()
        bar.update()

