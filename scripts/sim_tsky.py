# A script to create Tsky vs LST
from astropy.io import fits
from pygsm import GSMObserver
import numpy as np
from scipy import interpolate
from datetime import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
import healpy as hp
import glob

(latitude, longitude, elevation) = ('-30.7224', '21.4278', 1100)

hera_beam_file = '/data4/beards/instr_data/HERA_beam_nic.hmap'

df = 1.5625  # 100 MHz / 64 averaged channels
freqs = np.arange(100.0 + df / 2.0, 200.0, df)
hours = np.arange(0.0, 24.0, .5)
pols = ['X', 'Y']  # Only have X beam, but try rotating 90 degrees for Y

# Read in HERA beam data, just use full sky for paper
hera_beam = {}
# Only have X right now, will rotate later
hera_im = fits.getdata(hera_beam_file, extname='BEAM_{0}'.format('X'))
nside = hp.npix2nside(hera_im.shape[0])
f = lambda x,y,z: hp.pixelfunc.vec2pix(nside,x,y,z,nest=False)
temp_f = fits.getdata(hera_beam_file, extname='FREQS_{0}'.format('X'))
# Interpolate to the desired frequencies
func = interpolate.interp1d(temp_f, hera_im, kind='cubic', axis=1)
for pol in pols:
    hera_beam[pol] = func(freqs)

gsm_file = '/data4/tcox/HERA_IDR2_analysis/gsm.npz'
sky_array = np.load(gsm_file)['sky']

def HERA_Tsky(pols, freqs, return_sky = False, save_sky = False,
              Tsky_file = None, smoothing = False, deg = 1):
    lsts = np.zeros_like(hours)
    HERA_Tsky = np.zeros((len(pols), freqs.shape[0], lsts.shape[0]))
    for poli, pol in enumerate(pols):
        pol_ang = 90 * (1-poli)  # Extra rotation for X
        proj_beam = hp.projector.OrthographicProj(rot=[pol_ang,90], half_sky=True, xsize=400)
        for fi, freq in enumerate(freqs):
            if smoothing:
                deg_to_rad = np.pi / 180.0
                sig_to_fwhm = 2.4
                beam = hp.sphtfunc.smoothing(hera_beam[pol][:, fi],
                                             fwhm=deg_to_rad*sig_to_fwhm*deg)
            else:
                beam = hera_beam[pol][:, fi]
            print 'Forming HERA Tsky for frequency ' + str(freq) + ' MHz.'
            hbeam = proj_beam.projmap(beam, f)
            hbeam[np.isinf(hbeam)] = np.nan
            for ti, t in enumerate(hours):
                dt = datetime(2013, 1, 1, np.int(t), np.int(60.0 * (t - np.floor(t))),
                              np.int(60.0 * (60.0 * t - np.floor(t * 60.0))))
                lsts[ti] = Time(dt).sidereal_time('apparent', longitude).hour
                HERA_Tsky[poli, fi, ti] = np.nanmean(hbeam * sky_array[fi, ti, :, :]) / np.nanmean(hbeam)
    inds = np.argsort(lsts)
    lsts = lsts[inds]
    HERA_Tsky = HERA_Tsky[:, :, inds]

    if save_sky:
        np.savez(Tsky_file, HERA_Tsky=HERA_Tsky, freqs=freqs, lsts=lsts)
'''
for deg in np.linspace(1,12,10):
        HERA_Tsky(pols, freqs, smoothing = True, deg = deg,
                  Tsky_file = 'HERA_Tsky_{}_deg_conv.npz')
'''
HERA_Tsky(pols, freqs, save_sky = True, Tsky_file = '/data4/tcox/HERA_IDR2_analysis/HERA_Tsky_no_conv.npz')
