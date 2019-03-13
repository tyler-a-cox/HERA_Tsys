# A script to create Tsky vs LST

from astropy.io import fits
from pygsm import GSMObserver
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

# Set up the observer
(latitude, longitude, elevation) = ('-30.7224', '21.4278', 1100)
ov = GSMObserver()
ov.lon = longitude
ov.lat = latitude
ov.elev = elevation

d = ov.generate(100)
sky = hp.orthview(d, fig=0, xsize=400, return_projected_map=True,
                  half_sky=True)

g_sky = np.zeros((freqs.shape[0], hours.shape[0], sky.shape[0], sky.shape[1]))


fig = plt.figure("Tsky calc")

for fi, freq in enumerate(freqs):
    print 'Forming Tsky for frequency ' + str(freq) + ' MHz.'
    for ti, t in enumerate(hours):
        plt.clf()
        dt = datetime(2013, 1, 1, np.int(t), np.int(60.0 * (t - np.floor(t))),
                      np.int(60.0 * (60.0 * t - np.floor(t * 60.0))))
        lsts[ti] = Time(dt).sidereal_time('apparent', longitude).hour
        ov.date = dt
        ov.generate(freq)
        d = ov.view(fig=fig.number)
        sky = hp.orthview(d, fig=fig.number, xsize=400, return_projected_map=True,
                          half_sky=True)
        sky[np.isinf(sky)] = np.nan
        g_sky[fi, ti, :, :] = sky

Tsky_file = 'Tsky.npz'
np.savez(Tsky_file, sky=g_sky)
