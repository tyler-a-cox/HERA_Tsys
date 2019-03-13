from multiprocessing import Pool
import multiprocessing as mp
from astropy.io import fits
from pygsm import GSMObserver
import numpy as np
from scipy import interpolate
from datetime import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
import healpy as hp
import time

# hera_beam_file = '/data4/beards/instr_data/HERA_HFSS_X4Y2H_4900.hmap'
hera_beam_file = '/Users/tyler/Desktop/Research/Tsys/data/HERA_beam_nic.hmap'

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


proj_sky = hp.projector.OrthographicProj(rot=[0,0,0], half_sky=True, xsize=400)
f = lambda x,y,z: hp.pixelfunc.vec2pix(nside,x,y,z,nest=False)

def T_sky_calc(args):
    ov = GSMObserver()
    ov.lon = longitude
    ov.lat = latitude
    ov.elev = elevation
    fi, freq, poli, pol = args
    pol_ang = 90 * (1 - poli)  # Extra rotation for X
    print 'Forming HERA Tsky for frequency ' + str(freq) + ' MHz.'
    #smoothed_beam = hp.sphtfunc.smoothing(hera_beam[pol][:, fi], fwhm=0.017*deg)
    proj_beam = hp.projector.OrthographicProj(rot=[pol_ang,90], half_sky=True, xsize=400)
    hbeam = proj_beam.projmap(hera_beam[pol][:, fi], f)
    hbeam[np.isinf(hbeam)] = np.nan
    for ti, t in enumerate(hours):
        print freq, ti
        dt = datetime(2013, 1, 1, np.int(t), np.int(60.0 * (t - np.floor(t))),
                      np.int(60.0 * (60.0 * t - np.floor(t * 60.0))))
        lsts[ti] = Time(dt).sidereal_time('apparent', longitude).hour
        ov.date = dt
        observed_sky = ov.generate(freq)
        nside_sky = hp.pixelfunc.npix2nside(hp.pixelfunc.get_map_size(observed_sky))
        f_sky = lambda x,y,z: hp.pixelfunc.vec2pix(nside_sky,x,y,z, nest=False)
        sky = proj_sky.projmap(observed_sky, f_sky)
        sky[np.isinf(sky)] = np.nan
        HERA_Tsky[poli, fi, ti] = np.nanmean(hbeam * sky) / np.nanmean(hbeam)

    return HERA_Tsky


if __name__ == '__main__':
    poli, pol = [0, 'X']
    pool = Pool(processes=2)
    output = pool.map(T_sky_calc, [[fi,freq,poli,pol] for fi, freq in enumerate(freqs[:2])])
    '''
    for outi, out in enumerate(output):
        lsts = out[0]
        HERA_Tsky[poli, out[2], :] = out[1]
    '''

    inds = np.argsort(lsts)
    lsts = lsts[inds]
    HERA_Tsky = HERA_Tsky[:, :, inds]
    print (HERA_Tsky)
    #Tsky_file = 'HERA_tsky_fast.npz'
    #np.savez(Tsky_file, HERA_Tsky=HERA_Tsky, freqs=freqs, lsts=lsts)
