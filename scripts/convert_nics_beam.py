# Read Nic's beam output from CST and convert to HEALPix, store in a format
# that's easier to import

import numpy as np
import healpy as hp
from glob import glob
from astropy.io import fits
from scipy import interpolate
import re

nside = 64
subdir = True  # Nic originally had two directories of files, with very different formats
nicdir = '/data4/beards/instr_data/Simulations/Radiation patterns/'
if subdir:
    fileglob = nicdir + 'Directivity/farfield*.txt'
    filelist = np.array(glob(fileglob))
    freqs = np.array([float(re.findall(r'\d+', f.split()[-2])[0]) for f in filelist])
else:
    fileglob = nicdir + 'Directivity*MHz.txt'
    filelist = np.array(glob(fileglob))
    freqs = np.array([float(f.split()[-2]) for f in filelist])

order = np.argsort(freqs)
freqs = freqs[order]
filelist = filelist[order]

hmap = np.zeros((hp.nside2npix(nside), len(freqs)))

latitude = np.arange(181) * np.pi / 180.0
longitude = np.arange(360) * np.pi / 180.0
nlat = len(latitude)
nlong = len(longitude)
gain = np.zeros((nlat, nlong))

for fi, f in enumerate(filelist):
    data = np.loadtxt(f, skiprows=2)
    if subdir:
        nlat = len(np.unique(data[:, 0]))
        nlong = len(np.unique(data[:, 1]))
        latitude = (data[:, 0] * np.pi / 180.0)[0:nlat]
        longitude = (data[:, 1] * np.pi / 180.0)[0::nlat]
        gain = data[:, 2].reshape(nlong, nlat).transpose()
    else:
        gain_mixed = 10.0**(data[:, 2] / 10.0)
        data[:, 0:2] *= np.pi / 180.0
        data[data[:, 0] < 0, 1] += np.pi
        data[:, 1] += np.pi / 2
        data[:, 0] = np.abs(data[:, 0])
        for i in np.arange(len(data[:, 2])):
            lati = np.where(np.isclose(latitude, data[i, 0]))[0]
            loni = np.where(np.isclose(longitude, data[i, 1]))[0]
            gain[lati, loni] = gain_mixed[i]
        gain[0, :] = gain[0, 0]  # Fill in redundant values
        gain[-1, :] = gain[-1, -1]
    lut = interpolate.RectBivariateSpline(latitude, longitude, gain)
    thetai, phii = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    for i in np.arange(hp.nside2npix(nside)):
        hmap[i, fi] = lut(thetai[i], phii[i])

hmap /= hmap.max(axis=0, keepdims=True)

outfile = '/data4/beards/instr_data/HERA_beam_nic.hmap'

new_hdul = fits.HDUList()
new_hdul.append(fits.ImageHDU(data=hmap, name='BEAM_X'))
new_hdul.append(fits.ImageHDU(data=freqs, name='FREQS_X'))
new_hdul.writeto(outfile)
