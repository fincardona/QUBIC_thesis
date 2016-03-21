from __future__ import division

from qubic.io import write_map, read_map
from MyQubicInstrument import MyQubicInstrument
from pyspark import SparkContext
from pysimulators.interfaces.healpy import SceneHealpixCMB

import numpy as np
import matplotlib.pyplot as mp
import healpy as hp

def write(on, off, r, nside, idet, theta_max=45, syb_f=None, NPOINTS=1):
    path = '/home/fincardona/Qubic/spatial_extension/maps/mono'
    scene = SceneHealpixCMB(nside, kind='I')
    grid = np.concatenate([np.linspace(cut_on, cut_off, res) 
                           for cut_on, cut_off, res in zip(on, off, r)])
    for nu in grid:
        q = MyQubicInstrument(
            filter_nu=nu, filter_relative_bandwidth=1/nu, 
            synthbeam_dtype=float, synthbeam_fraction=syb_f)
        sb = q.get_synthbeam(scene, idet, theta_max, NPOINTS=NPOINTS)
        sb_direct_ga = q.direct_convolution(
            scene, idet, theta_max, NPOINTS=NPOINTS)
        write_map(
            '%s/interfero/sb_nside{}_nu{:.3e}_idet{}_tmax{}_Npoints{}.fits'.
            format(nside, nu, idet, theta_max, NPOINTS) % path, 
            np.sum(sb, axis=0))
        write_map(
            '%s/direct_conv/dc_nside{}_nu{:.3e}_idet{}_tmax{}_sybf{}_Npoints{}.fits'.format(nside, nu, idet, theta_max, syb_f, 
            NPOINTS) % path, np.sum(sb_direct_ga, axis=0))

if __name__ == "__main__":

    # Parameters
    nside = 1024
    idet = 231
    syb_f = 0.99
    theta_max = 30

    NPOINTS = [4, 9, 16, 25, 36]
    
    on = [130e9, 190e9]
    off = [170e9, 250e9]
    r = [401, 601]
    
    # Synthesized Beams
    sc = SparkContext(appName="WriteMultiMonoBeams")
    sc.parallelize(NPOINTS).map(
        lambda x: write(on, off, r, nside, idet, theta_max, syb_f, 
                        x)).collect()
    
    sc.stop()
