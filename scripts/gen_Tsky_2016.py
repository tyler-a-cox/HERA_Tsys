# A script to create Tsky vs LST

from astropy.io import fits
from pygsm import GSMObserver2016
import numpy as np
from scipy import interpolate
from datetime import datetime
from astropy.time import Time
import matplotlib.pyplot as plt
import healpy as hp

df = 1.5625  # 100 MHz / 64 averaged channels
freqs = np.arange(100.0 + df / 2.0, 200.0, df)
hours = np.arange(0.0, 24.0, .5)
lsts = np.zeros_like(hours)
pols = ['X', 'Y']  # Only have X beam, but try rotating 90 degrees for Y
HERA_Tsky = np.zeros((len(pols), freqs.shape[0], lsts.shape[0]))

# Set up the observer
(latitude, longitude, elevation) = ('-30.7224', '21.4278', 1100)
ov = GSMObserver2016()
ov.lon = longitude
ov.lat = latitude
ov.elev = elevation

proj_sky = hp.projector.OrthographicProj(rot=[0,0,0], half_sky=True, xsize=400)

observed_sky = ov.generate(100)

nside_sky = hp.pixelfunc.npix2nside(hp.pixelfunc.get_map_size(observed_sky))
f_sky = lambda x,y,z: hp.pixelfunc.vec2pix(nside_sky, x, y, z, nest=False)
sky = proj_sky.projmap(observed_sky, f_sky)

generated_sky = np.zeros((freqs.shape[0], hours.shape[0], sky.shape[0],
                          sky.shape[1]))

print generated_sky.shape

for fi, freq in enumerate(freqs):
    print 'Forming Tsky for frequency ' + str(freq) + ' MHz.'
    for ti, t in enumerate(hours):
        dt = datetime(2013, 1, 1, np.int(t), np.int(60.0 * (t - np.floor(t))),
                      np.int(60.0 * (60.0 * t - np.floor(t * 60.0))))
        lsts[ti] = Time(dt).sidereal_time('apparent', longitude).hour
        ov.date = dt
        observed_sky = ov.generate(freq)
        sky = proj_sky.projmap(observed_sky, f_sky)
        sky[np.isinf(sky)] = np.nan
        sky = np.flip(sky, axis=0)
        generated_sky[fi, ti, :, :] = sky

np.savez('gsm',sky = generated_sky)
