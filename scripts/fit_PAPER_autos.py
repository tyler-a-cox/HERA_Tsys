# Fit autos to Tsky template
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt


# Define curve to fit data to
def curve_to_fit(lsts, gain, gain_slope, rxr_amp, rxr_slope):
    global interp_values
    lsts_shifted = lsts - lsts.mean()
    return (gain * interp_values + gain_slope * lsts_shifted * interp_values +
            rxr_amp + rxr_slope * lsts_shifted)


def match_model_to_data(lsts, fi, params):
    global interp_func
    interp_values = interp_func(lsts)[0, fi, :]
    lsts_shifted = lsts - lsts.mean()
    return (params[0] * interp_values + params[1] * lsts_shifted * interp_values +
            params[2] + params[3] * lsts_shifted)


def match_data_to_model(lsts, data, params):
    lsts_shifted = lsts - lsts.mean()
    return ((data - params[2] - params[3] * lsts_shifted) /
            (params[0] + params[1] * lsts_shifted))


HERA_hex_list = [9, 10, 20, 22, 31, 43, 53, 64, 65, 72, 80, 81, 88, 89, 96, 97, 104, 105, 112]
PAPER_hex_list = [0, 2, 14, 17, 21, 40, 44, 45, 54, 62, 68, 69, 84, 85, 86, 100, 101, 102, 113]
PAPER_imaging_list = [1, 3, 4, 13, 15, 16, 23, 26, 37, 38, 41, 42, 46, 47, 49,
                      50, 56, 57, 58, 59, 61, 63, 66, 67, 70, 71, 73, 74, 82,
                      83, 87, 90, 98, 99, 103, 106, 114, 115, 116, 117, 118,
                      119, 120, 121, 122, 123, 124, 125, 126, 127]
PAPER_pol_list = [5, 6, 7, 8, 11, 12, 18, 19, 24, 25, 27, 28, 29, 30, 32, 33,
                  34, 35, 36, 39, 48, 51, 52, 55, 60, 75, 76, 77, 78, 79, 91,
                  92, 93, 94, 95, 107, 108, 109, 110, 111]
PAPER_list = PAPER_hex_list + PAPER_imaging_list + PAPER_pol_list
# Load data from Tsky_v_LST.py
Tsky_file = '/data2/beards/tmp/PAPER_Tsky.npz'
data = np.load(Tsky_file)
freqs = data['freqs']
model_lsts = data['lsts']
PAPER_Tsky = data['PAPER_Tsky']

PAPER_fits = np.zeros((len(PAPER_list), len(freqs), 4))  # sky_amp, sky_slope, rxr_amp, rxr_slope
interp_func = interpolate.interp1d(model_lsts, PAPER_Tsky, kind='cubic', axis=2)
for fi, freq in enumerate(freqs):
    interp_values = interp_func(lsts)[0, fi, :]
    for anti, ant in enumerate(PAPER_list):
        PAPER_fits[anti, fi, :] = curve_fit(curve_to_fit, lsts, xxd_ave[ant, :, fi])[0]

# Get gains and receiver temperatures
PAPER_gains = np.zeros((len(PAPER_list), len(freqs), len(lsts)))
PAPER_rxr_temp = np.zeros((len(PAPER_list), len(freqs), len(lsts)))
lsts_shifted = lsts - lsts.mean()
for anti, ant in enumerate(PAPER_list):
    PAPER_gains[anti, :, :] = (PAPER_fits[anti, :, 0].reshape(-1, 1) +
                               PAPER_fits[anti, :, 1].reshape(-1, 1) *
                               lsts_shifted.reshape(1, -1))
    PAPER_rxr_temp[anti, :, :] = (PAPER_fits[anti, :, 2].reshape(-1, 1) +
                                  PAPER_fits[anti, :, 3].reshape(-1, 1) *
                                  lsts_shifted.reshape(1, -1)) / PAPER_gains[anti, :, :]

# Do some plotting
# first plot some of the actual fits
outdir = '/data2/beards/PAPER_auto_data/'
finds = [10, 32, 54]  # beginning, middle, end, but not at the very edges
fit_fig = plt.figure('PAPER auto fits')
for anti, ant in enumerate(PAPER_list):
    fit_fig.clf()
    for fi in finds:
        temp, = plt.plot(lsts, match_data_to_model(lsts, xxd_ave[ant, :, fi],
                                                   PAPER_fits[anti, fi, :]), '.', ms=5)
        plt.plot(lsts, interp_func(lsts)[0, fi, :], '.', ms=2)
    ylim([0, 1.3 * np.max(interp_func(lsts)[0, finds[0], :])])
    xlabel('LST (Hours)')
    ylabel('Tsky')
    title('Antenna ' + str(ant))
    outfile = outdir + 'Tsky_v_LST_fit' + str(ant) + '.png'
    savefig(outfile)

gains_fig = plt.figure('PAPER gains')
for anti, ant in enumerate(PAPER_list):
    gains_fig.clf()
    imshow(PAPER_gains[anti, :, :].transpose(), aspect='auto', origin='lower',
           extent=(freqs.min(), freqs.max(), lsts.min(), lsts.max()), interpolation='none')
    xlabel('Frequency (MHz)')
    ylabel('LST (Hour)')
    title('Rough gain, Antenna ' + str(ant))
    colorbar()
    outfile = outdir + 'gain_waterfall' + str(ant) + '.png'
    savefig(outfile)

rxr_fig = plt.figure('PAPER gains')
for anti, ant in enumerate(PAPER_list):
    rxr_fig.clf()
    imshow(np.log10(PAPER_rxr_temp[anti, :, :]).transpose(), aspect='auto', origin='lower',
           extent=(freqs.min(), freqs.max(), lsts.min(), lsts.max()), interpolation='none')
    xlabel('Frequency (MHz)')
    ylabel('LST (Hour)')
    title('Rxr temperature (log10(K)), Antenna ' + str(ant))
    clim([0, 2 * np.log10(np.median(PAPER_rxr_temp[anti, :, :]))])
    colorbar()
    outfile = outdir + 'rxr_temp_waterfall' + str(ant) + '.png'
    savefig(outfile)
