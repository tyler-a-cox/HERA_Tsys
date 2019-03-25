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


# hera_beam_file = '/data4/beards/instr_data/HERA_HFSS_X4Y2H_4900.hmap'
hera_beam_file = '/Users/tyler/Desktop/Research/Tsys/data/HERA_beam_nic.hmap'
sky_files = glob.glob('sky_data/*.npz')
sky_files.sort()

df = 1.5625  # 100 MHz / 64 averaged channels
freqs = np.arange(100.0 + df / 2.0, 200.0, df)
hours = np.arange(0.0, 24.0, .5)
lsts = np.zeros_like(hours)
pols = ['X', 'Y']  # Only have X beam, but try rotating 90 degrees for Y
HERA_Tsky = np.zeros((len(pols), freqs.shape[0], lsts.shape[0]))

# Read in HERA beam data, just use full sky for paper
hera_beam = {}
# Only have X right now, will rotate later
hera_im = fits.getdata(hera_beam_file, extname='BEAM_{0}'.format('X'))
nside = hp.npix2nside(hera_im.shape[0])
temp_f = fits.getdata(hera_beam_file, extname='FREQS_{0}'.format('X'))
# Interpolate to the desired frequencies
func = interpolate.interp1d(temp_f, hera_im, kind='cubic', axis=1)
for pol in pols:
    hera_beam[pol] = func(freqs)

# Set up the observer
(latitude, longitude, elevation) = ('-30.7224', '21.4278', 1100)
ov = GSMObserver()
ov.lon = longitude
ov.lat = latitude
ov.elev = elevation

f = lambda x,y,z: hp.pixelfunc.vec2pix(nside,x,y,z,nest=False)
i = 0
j = 0
sky_array = np.load(sky_files[i])['sky']

print sky_files

for poli, pol in enumerate(pols):
    pol_ang = 90 * (1-poli)  # Extra rotation for X
    proj_beam = hp.projector.OrthographicProj(rot=[pol_ang,90], half_sky=True, xsize=400)
    for fi, freq in enumerate(freqs):
        #deg = 8
        #smoothed_beam = hp.sphtfunc.smoothing(hera_beam[pol][:, fi], fwhm=0.017*deg)
        print 'Forming HERA Tsky for frequency ' + str(freq) + ' MHz.'
        hbeam = proj_beam.projmap(hera_beam[pol][:, fi], f)
        hbeam[np.isinf(hbeam)] = np.nan
        for ti, t in enumerate(hours):
            dt = datetime(2013, 1, 1, np.int(t), np.int(60.0 * (t - np.floor(t))),
                          np.int(60.0 * (60.0 * t - np.floor(t * 60.0))))
            lsts[ti] = Time(dt).sidereal_time('apparent', longitude).hour
            HERA_Tsky[poli, fi, ti] = np.nanmean(hbeam * sky_array[j, ti, :, :]) / np.nanmean(hbeam)
        j = (fi+1) % 8
        if (fi+1) % 8 == 0 and i < 7:
            i += 1
            sky_array = np.load(sky_files[i])['sky']
        elif i >= 7 and poli == 0:
            i = 0
            sky_array = np.load(sky_files[i])['sky']

inds = np.argsort(lsts)
lsts = lsts[inds]
HERA_Tsky = HERA_Tsky[:, :, inds]

Tsky_file = 'HERA_tsky_fast_1.npz'
np.savez(Tsky_file, HERA_Tsky=HERA_Tsky, freqs=freqs, lsts=lsts)
