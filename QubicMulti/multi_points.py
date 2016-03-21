from __future__ import division

from MyQubicInstrument import MyQubicInstrument

import numpy as np
import healpy as hp
import matplotlib.pyplot as mp

def square_grid(NPOINTS, side):
    n = np.int(np.sqrt(NPOINTS))
    a = np.linspace(0, 1, 2 * n + 1)[range(1, 2 * n + 1, 2)]
    x, y = np.meshgrid(a, a)
    return np.array(zip(x.ravel(), y.ravel())) * side

def shift_grid(NPOINTS, side, central_position):
    grid_ = square_grid(NPOINTS, side)
    grid = np.full((len(central_position), len(grid_), 2), grid_)
    points = grid + (
        central_position[...,:-1] - np.mean(grid_, axis=0))[:,None,:]
    return np.concatenate((
        points, np.full_like(points[...,0,None], -0.3)), axis=-1)

def npoints(NPOINTS, idet):
    q = MyQubicInstrument()
    side = np.sqrt(q.detector.area)
    central_pos = q[idet].detector.center
    vertex = q[idet].detector.vertex
    return shift_grid(NPOINTS, side, central_pos)

def plot_points(points):
    q = MyQubicInstrument()
    central_pos = q[idet].detector.center
    vertex = q[idet].detector.vertex
    mp.figure()
    #mp.plot(vertex[...,0], vertex[...,1], 'ro')
    if points.shape[1] != 1:
        mp.plot(points[..., 0], points[..., 1], 'bx')
    mp.plot(central_pos[..., 0], central_pos[..., 1], 'y*')
    q[idet].detector.plot()
    mp.show()

idet = np.arange(248)
NPOINTS = 36
points = npoints(NPOINTS, idet)
plot_points(points)

