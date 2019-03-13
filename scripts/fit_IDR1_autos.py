# Fit autos to Tsky template
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt


# Define curve to fit data to
def curve_to_fit(lsts, gain, gain_slope, rxr_amp, rxr_slope):
    global interp_values
    lsts_shifted = lsts - lsts.min()
    return (gain * interp_values + gain_slope * lsts_shifted * interp_values +
            rxr_amp + rxr_slope * lsts_shifted)


def match_model_to_data(lsts, pol, fi, params):
    global interp_func
    interp_values = interp_func(lsts)[pol, fi, :]
    lsts_shifted = lsts - lsts.min()
    return (params[0] * interp_values + params[1] * lsts_shifted * interp_values +
            params[2] + params[3] * lsts_shifted)


def match_data_to_model(lsts, data, params):
    lsts_shifted = lsts - lsts.min()
    return ((data - params[2] - params[3] * lsts_shifted) /
            (params[0] + params[1] * lsts_shifted))

HERA_list = [9, 10, 20, 22, 31, 43, 53, 64, 65, 72, 80, 81, 88, 89, 96, 97, 104, 105, 112]
PAPER_hex_list = [0, 2, 14, 17, 21, 40, 44, 45, 54, 62, 68, 69, 84, 85, 86, 100, 101, 102, 113]
PAPER_imaging_list = [1, 3, 4, 13, 15, 16, 23, 26, 37, 38, 41, 42, 46, 47, 49,
                      50, 56, 57, 58, 59, 61, 63, 66, 67, 70, 71, 73, 74, 82,
                      83, 87, 90, 98, 99, 103, 106, 114, 115, 116, 117, 118,
                      119, 120, 121, 122, 123, 124, 125, 126, 127]
PAPER_pol_list = [5, 6, 7, 8, 11, 12, 18, 19, 24, 25, 27, 28, 29, 30, 32, 33,
                  34, 35, 36, 39, 48, 51, 52, 55, 60, 75, 76, 77, 78, 79, 91,
                  92, 93, 94, 95, 107, 108, 109, 110, 111]
PAPER_list = PAPER_hex_list + PAPER_imaging_list + PAPER_pol_list
nant = len(HERA_list) + len(PAPER_list)
pols = ['X', 'Y']
npol = 2

# Load data from Tsky_v_LST.py
Tsky_file = '/data2/beards/tmp/HERA_Tsky.npz'
data = np.load(Tsky_file)
freqs = data['freqs']
model_lsts = data['lsts']
HERA_Tsky = data['HERA_Tsky']
Tsky_file = '/data2/beards/tmp/PAPER_Tsky.npz'
data = np.load(Tsky_file)
PAPER_Tsky = data['PAPER_Tsky']

auto_fits = np.zeros((npol, nant, len(freqs), 4))  # sky_amp, sky_slope, rxr_amp, rxr_slope
h_interp_func = interpolate.interp1d(model_lsts, HERA_Tsky, kind='cubic', axis=2)
p_interp_func = interpolate.interp1d(model_lsts, PAPER_Tsky, kind='cubic', axis=2)
for pol in xrange(npol):
    for fi, freq in enumerate(freqs):
        interp_values = h_interp_func(lsts[pol])[pol, fi, :]
        for ant in HERA_list:
            auto_fits[pol, ant, fi, :] = curve_fit(curve_to_fit, lsts[pol],
                                                   data_ave[pol][ant, :, fi])[0]
        interp_values = p_interp_func(lsts[pol])[pol, fi, :]
        for ant in PAPER_list:
            auto_fits[pol, ant, fi, :] = curve_fit(curve_to_fit, lsts[pol],
                                                   data_ave[pol][ant, :, fi])[0]

# Get gains and receiver temperatures
gains = np.zeros((npol, nant, len(freqs), len(lsts[0])))
rxr_temp = np.zeros((npol, nant, len(freqs), len(lsts[0])))
lsts_shifted = [lsts[0] - lsts[0].min(), lsts[1] - lsts[1].min()]
for pol in xrange(npol):
    for ant in xrange(nant):
        gains[pol, ant, :, :] = (auto_fits[pol, ant, :, 0].reshape(-1, 1) +
                                 auto_fits[pol, ant, :, 1].reshape(-1, 1) *
                                 lsts_shifted[pol].reshape(1, -1))
        rxr_temp[pol, ant, :, :] = (auto_fits[pol, ant, :, 2].reshape(-1, 1) +
                                    auto_fits[pol, ant, :, 3].reshape(-1, 1) *
                                    lsts_shifted[pol].reshape(1, -1)) / gains[pol, ant, :, :]

# Do some plotting
# first plot some of the actual fits
outdir = '/data2/beards/IDR1_auto_data/'
finds = [10, 32, 54]  # beginning, middle, end, but not at the very edges
dark_colors = ['blue', 'green', 'red']
light_colors = ['dodgerblue', 'lightgreen', 'salmon']
fig = plt.figure('auto fits')
for pol in xrange(npol):
    for ant in xrange(nant):
        # Fits plot
        fig.clf()
        if ant in HERA_list:
            tittext = 'Antenna ' + str(ant) + pols[pol] + ' (HERA)'
            interp_func = h_interp_func
        elif ant in PAPER_hex_list:
            tittext = 'Antenna ' + str(ant) + pols[pol] + ' (PAPER hex)'
            interp_func = p_interp_func
        elif ant in PAPER_pol_list:
            tittext = 'Antenna ' + str(ant) + pols[pol] + ' (PAPER pol)'
            interp_func = p_interp_func
        else:
            tittext = 'Antenna ' + str(ant) + pols[pol] + ' (PAPER imaging)'
            interp_func = p_interp_func

        for i, fi in enumerate(finds):
            label = str(freqs[fi]) + ' MHz'
            plt.plot(lsts[pol], match_data_to_model(lsts[pol], data_ave[pol][ant, :, fi],
                                                    auto_fits[pol, ant, fi, :]), '.',
                     ms=5, color=dark_colors[i], label=label)
            plt.plot(lsts[pol], interp_func(lsts[pol])[pol, fi, :],
                     color=light_colors[i])
        ylim([0, 1.3 * np.max(interp_func(lsts[pol])[pol, finds[0], :])])
        xlabel('LST (Hours)')
        ylabel('Tsky')
        title(tittext)
        legend()
        outfile = outdir + 'Tsky_v_LST_fit' + str(ant) + pols[pol] + '.png'
        savefig(outfile)

        # Gain plot
        fig.clf()
        imshow(gains[pol, ant, :, :].transpose(), aspect='auto', origin='lower',
               extent=(freqs.min(), freqs.max(), lsts[pol].min(), lsts[pol].max()),
               interpolation='none')
        xlabel('Frequency (MHz)')
        ylabel('LST (Hour)')
        title(tittext)
        colorbar(label='Gain')
        outfile = outdir + 'gain_waterfall' + str(ant) + pols[pol] + '.png'
        savefig(outfile)

        fig.clf()
        imshow(np.log10(rxr_temp[pol, anti, :, :]).transpose(), aspect='auto', origin='lower',
               extent=(freqs.min(), freqs.max(), lsts[pol].min(), lsts[pol].max()),
               interpolation='none')
        xlabel('Frequency (MHz)')
        ylabel('LST (Hour)')
        title(tittext)
        clim([0, 3 * np.log10(np.median(rxr_temp[pol, anti, :, :]))])
        colorbar(label='Rxr Temp, Log10(K)')
        outfile = outdir + 'rxr_temp_waterfall' + str(ant) + pols[pol] + '.png'
        savefig(outfile)

# Final plot is gain vs Antenna
mean_gains = np.mean(gains, axis=(2, 3))
fig.clf()
syms = ['x', 'v']
handles = []
temp, = plt.plot(HERA_list, mean_gains[0, HERA_list], 'x', label='HERA', mew=2.5, ms=6)
handles += [temp]
plt.plot(HERA_list, mean_gains[1, HERA_list], 'v', color=temp.get_color())
temp, = plt.plot(PAPER_hex_list, mean_gains[0, PAPER_hex_list], 'x', label='PAPER hex', mew=2.5, ms=6)
handles += [temp]
plt.plot(PAPER_hex_list, mean_gains[1, PAPER_hex_list], 'v', color=temp.get_color())
temp, = plt.plot(PAPER_pol_list, mean_gains[0, PAPER_pol_list], 'x', label='PAPER pol', mew=2.5, ms=6)
handles += [temp]
plt.plot(PAPER_pol_list, mean_gains[1, PAPER_pol_list], 'v', color=temp.get_color())
temp, = plt.plot(PAPER_imaging_list, mean_gains[0, PAPER_imaging_list], 'x', label='PAPER imaging', mew=2.5, ms=6)
handles += [temp]
plt.plot(PAPER_imaging_list, mean_gains[1, PAPER_imaging_list], 'v', color=temp.get_color())
ylim([-.01, .05])
xlabel('Antenna number')
ylabel('Average Gain')
plt.legend(handles=handles)
outfile = outdir + 'avg_gains.png'
savefig(outfile)
