from __future__ import division

from MultiQubicAcquisition import MultiQubicAcquisition
from MultiQubicInstrument import MultiQubicInstrument
from MultiQubicScene import MultiQubicScene
from qubic import create_random_pointings
from qubic.io import read_map
from qubic import read_spectra
from scipy.constants import c

import matplotlib.pyplot as mp
import qubic_sensitivity as qs
import healpy as hp
import numpy as np

def cmb_power_spectrum(ell, r=0.1, kind='BB'):
    lmin, lmax = (ell[0], ell[-1])
    if kind == 'TT':
        return read_spectra(r)[0][lmin: lmax + 1] 
    elif kind == 'EE':
        return read_spectra(r)[1][lmin: lmax + 1]
    elif kind == 'BB':
        return read_spectra(r)[2][lmin: lmax + 1]
    elif kind == 'TE':
        return read_spectra(r)[3][lmin: lmax + 1]
    else: 
        raise ValueError("kind must be: TT, EE, BB or TE") 

def cl2dl(Cl, ell):
    return Cl * ell * (ell + 1) / 2 / np.pi 

def cl_dl(ell, r=0.05):
    Cl = cmb_power_spectrum(ell, r=r) 
    return Cl, cl2dl(Cl, ell)

def equivalent_baselines(instrument, ell, f_sky, hamilton=True):
    N_h = len(instrument.horn)
    if hamilton == False:
        # return the formula 8.47 of the Romain Charlassier's thesis
        return N_h, (1 - (np.sqrt(2) / 2 / np.sqrt(N_h)) - 
                     (ell * np.sqrt(f_sky) / 2 / np.sqrt(N_h))) * N_h
    ellbs_nbs = qs.give_baselines(instrument, freqHz=instrument.filter.name)
    a, b, c = np.polyfit(ellbs_nbs[-2], ellbs_nbs[-1], 2)
    return N_h, a * ell**2 + b * ell + c

def integrate_k_one(q, ell, omega):
    w = q.filter.bandwidth 
    nu0, nu = (q.filter.name, q.filter.nu)
    I = np.zeros((len(ell), q.filter.NFREQS))
    for i, l in enumerate(ell):
        I[i] = omega * np.exp(-omega * l**2 / 4 / np.pi * ((nu - nu0) / nu0) 
                              **2) * w / np.sum(w)
    return np.sum(I, axis=1)

def k_one(instr, ell, omega, hamilton=True):
    if hamilton == False:
        sigma_l = np.sqrt(2 * np.pi / omega) # 1/instr.primary_beam.sigma
        delta_nu_over_nu = np.sum(instr.filter.relative_bandwidth)
        return np.sqrt(1 + (delta_nu_over_nu * ell / sigma_l)**2)
    return omega / integrate_k_one(instr, ell, omega)

def wpixel(instr, ell, nside=1024, idet=231, theta_max=30):
    nu = instr.filter.name
    NPOINTS = instr.detector.NPOINTS
    if NPOINTS == 1:
        return np.ones(len(ell))
    path_ = '/home/federico/Desktop/MAPS/no_spat_ext/poly/interfero'
    pathN = '/home/federico/Desktop/MAPS/spat_ext/poly/interfero'
    beam_ = read_map('%s/sb_nside{}_mono_{}Ghz_idet{}_tmax{}.fits'.
                format(nside, int(nu/1e9), idet, theta_max) % path_)
    beamN = read_map('%s/sb_nside{}_mono_{}Ghz_idet{}_tmax{}_Npoints{}.fits'.
                format(nside, int(nu/1e9), idet, theta_max, NPOINTS) % pathN)
    return (hp.anafast(beamN, lmax=ell[-1], pol=False) / 
            hp.anafast(beam_, lmax=ell[-1], pol=False))[ell[0]:]

def battistelli(
        ell, delta_ell, k1, f_sky, Cl, apodization, N_h, NET, omega, N_eq, t,
        w_pix, optical_eff, only_noise=False, kind='Interferometer'):
    a = np.sqrt(2 * k1 / (2 * ell + 1) / f_sky / delta_ell)
    if kind == 'Interferometer':
        b = (apodization * N_h * NET**2 * omega * k1 / N_eq**2 / t / w_pix / 
             optical_eff) 
        # miss the factor 2: for Battistelli fsky = 2 * np.pi * omega
    elif kind == 'Imager':
        b = 4 * np.pi * 1.14685569e-06 / w_pix / time * 8640 * 3650 
        # see J_Kaplan/Clbruit.py line 176-177
    else: 
        raise ValueError('kind must be: Imager or Interferometer')
    if only_noise == True:
        return a * b
    return a * (Cl + b)

def delta_dl(
        q, ell, delta_ell, f_sky, apodization, N_h, NET, omega, N_eq, t, 
        optical_eff, r=0.05, only_noise=False, kind='Interferometer'):
    Cl, Dl = cl_dl(ell, r=r)
    k1 = k_one(q, ell, omega)
    w_pix = wpixel(q, ell)
    delta_Cl = battistelli(
        ell, delta_ell, k1, f_sky, Cl, apodization, N_h, NET, omega, N_eq, t,
        w_pix, optical_eff, only_noise=only_noise, kind=kind) 
    return cl2dl(delta_Cl, ell)

def parameters(q, sampling, scene, maxiter, who='kaplan'):
    if who == 'kaplan':
        f_sky = 0.0096912
        apodization_factor = 1.27895995 
    elif who == 'me':
        PATH = '/home/federico/Dropbox/Tesi/Python/QubicWork'
        path = PATH + '/map_making/maps/montecarlo'
        cov = hp.read_map(
            '%s/qcoverage_n{}_N{}_s{}_mi{}.fits'.format(
            q.filter.NFREQS, q.detector.NPOINTS, len(sampling), 
            maxiter) % path)
        npixel = 12 * scene.nside**2
        mask = cov > cov.max() * 0.2
        f_sky = np.sum(mask) / npixel
        apodization_factor = np.sum(mask) / np.sum(mask**2)
        hp.write_map('maschera.fits', mask)
    else: 
        raise ValueError("who must be: kaplan or me ")
    omega = 4 * np.pi * f_sky
    return f_sky, omega, apodization_factor

nside = 256
npixel = 12 * nside ** 2 
scene = MultiQubicScene(nside)

racenter = 0.0      # deg
deccenter = -57.0   # deg
sampling = create_random_pointings([racenter, deccenter], 1000, 10)
maxiter = 1000

years = 2
time = 60 * 60 * 24 * 365 * years
detector_nep = 4.7e-17 
detector_noise = detector_nep * np.sqrt(
    len(sampling) * sampling.period / time)

r = 0.1
lmin = 2
lmax = nside * 2
ell = np.arange(lmin, lmax + 1)
delta_ell = 20

NFREQS = 1
NPOINTS = [1, 4, 9, 16, 25, 36]

Cl, Dl = cl_dl(ell, r=r)
mp.figure()
mp.loglog(ell, Dl, label='$D_\ell^{BB}\ $')

color = ['g', 'r', 'c', 'm', 'y', 'b', 'k']
color = color[0: len(NPOINTS)]

delta_Dls_tot = []
delta_Dls_xpol_tot = []
for NP, c in zip(NPOINTS, color):
    q = MultiQubicInstrument(
        NFREQS=NFREQS, NPOINTS=NP, detector_nep=detector_nep) 

    # Battistelli
    f_sky, omega, apodization_factor = parameters(
        q, sampling, scene, maxiter, who='me')
    N_h, N_eq = equivalent_baselines(q, ell, f_sky, hamilton=True)
    optical_efficiency = 1 
    NET = 595 # See J_kaplan/Clbruit.py
        
    delta_Dls = delta_dl(
        q, ell, delta_ell, f_sky, apodization_factor, N_h, NET, 
        omega, N_eq, time, optical_efficiency, r=r, only_noise=True,
        kind='Interferometer')
    delta_Dls_tot.append(delta_Dls)
    #Comparison with montecarlos
    #Xpol
    xpols = np.load(
        '/home/federico/Dropbox/Tesi/Python/QubicWork/map_making/Xpol/delta_Cls_' + 
        'n{}_N{}_bin{}_{}MAPS_s{}_mi{}.npy'.format(NFREQS, NP, 
        delta_ell, 100, len(sampling), maxiter))
    ell_binned_xpol = xpols[0]
    delta_Cls_xpol = xpols[1]
    delta_Dls_xpol = cl2dl(delta_Cls_xpol, ell_binned_xpol)
    delta_Dls_xpol_tot.append(delta_Dls_xpol[2])
    #Display
    mp.plot(ell, delta_Dls, '--' + c,
            label='$\Delta D_\ell^{BB}\ $' + 
            'Battistelli: n = {}; $N_P$ = {}'.
            format(q.filter.NFREQS, q.detector.NPOINTS))
    mp.plot(ell_binned_xpol, delta_Dls_xpol[2], '-v' + c,
            label='$\Delta D_\ell^{BB}\ $' + 
            'Xpol: n = {}; $N_P$ = {}'.format(
            q.filter.NFREQS, q.detector.NPOINTS))


mp.xlim(40, 300)
mp.xticks(fontsize='x-large')
mp.ylim([1e-4, 1e-1])
mp.yticks(fontsize='x-large')
mp.xlabel('$\ell$', fontsize='xx-large')
mp.ylabel('$[ {\mu k}^2 ]$', fontsize='xx-large')
mp.title('r = {}'.format(r), fontsize='x-large')
mp.legend(loc='upper left', bbox_to_anchor=(0, 1.14))


diff_monte = ((delta_Dls_xpol_tot - delta_Dls_xpol_tot[0]) / 
              delta_Dls_xpol_tot[0])
mean_diff_monte = np.mean(diff_monte, axis=1)

diff_theo = ((delta_Dls_tot - delta_Dls_tot[0]) / delta_Dls_tot[0])
mean_diff_theo = np.mean(diff_theo, axis=1)

mp.figure()
mp.title('Mean Difference $\ell $ by $\ell $; r = {}'.format(r), 
         fontsize='x-large')
mp.xticks(fontsize='x-large')
mp.yticks(fontsize='x-large')
mp.xlabel('$N_P$', fontsize='xx-large')
mp.ylabel(r'$\frac{{< \Delta D_\ell^{BB} (N_P)\ - \Delta D_\ell^{BB} (1)\ >}_\ell \ }{\Delta D_\ell^{BB} (1)\ } $', fontsize='xx-large')
mp.plot(NPOINTS, mean_diff_monte, '-o',label='Monte Carlo')
mp.plot(NPOINTS, mean_diff_theo, '-o',label='Theoretical')
mp.legend(loc='best')

mp.show()
