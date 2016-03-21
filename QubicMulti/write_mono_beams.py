from __future__ import division

from qubic.io import write_map
from MyQubicInstrument import MyQubicInstrument
from pyspark import SparkContext
from pysimulators.interfaces.healpy import SceneHealpixCMB

import numpy as np

def write(on, off, r, nside, idet, theta_max=45, syb_f=None):
    path = '/home/fincardona/Qubic/Compare_poly/maps/mono'
    scene = SceneHealpixCMB(nside, kind='I')
    grid = np.concatenate([np.linspace(cut_on, cut_off, res) 
                           for cut_on, cut_off, res in zip(on, off, r)])
    sb = np.empty(12*nside**2)
    for nu in grid:
        q = MyQubicInstrument(
            filter_nu=nu, filter_relative_bandwidth=1/nu, 
            synthbeam_dtype=float, synthbeam_fraction=syb_f)
        sb = q.get_synthbeam(scene, idet, theta_max)
        sb_direct_ga = q.direct_convolution(scene, idet, theta_max)
        write_map(
            '%s/interfero/sb_nside{}_nu{:.3e}_idet{}_tmax{}.fits'.
            format(nside, nu, idet, theta_max) % path, sb)
        write_map(
            '%s/direct_conv/dc_nside{}_nu{:.3e}_idet{}_tmax{}_sybf{}.fits'.
            format(nside, nu, idet, theta_max, syb_f) % path, sb_direct_ga)

if __name__ == "__main__":

    # Parameters
    nside = 1024
    
    det = [348, 419, 495, 496, 503, 624, 727, 744, 751, 844, 975]
    
    syb_f = 0.99
    theta_max = 30
    
    on = [130e9, 190e9]
    off = [170e9, 250e9]
    r = [401, 601]
    
    # Synthesized Beams
    sc = SparkContext(appName="WriteMonoBeams")
    sc.parallelize(det).map(
        lambda x: write(on, off, r, nside, x, theta_max, syb_f)).collect()
    
    sc.stop()
