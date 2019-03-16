# A script to get absolute gain factor for the HERA dishes from IDR1.
# Must run HERA_IDR1_read_autos.py and Tsky_v_LST.py first.
#
# I'm changing the function to fit to to make them more "orthogonal", and simplify
# by removing slopes.
# Old function to fit: data = y(LST) = g * Tsky(LST) + y0
# New function to fit: y'(LST) = g' * (Tsky(LST) - <Tsky>) + y0'
#     Connection between fits: g = g', y0 = y0' - g' * <Tsky>
#     Connection to real world: g = gain^2 to get into Temp units (K)
#                               Trxr = y0 / g = y0' / g' - <Tsky>

import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import aipy
from astropy.io import fits
from scipy.stats import chisquare
from sklearn.metrics import r2_score

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def curve_to_fit(lsts, gain, offset):
    # Define curve to fit data to
    # offset is Tsys offset from zero = Trxr + g * <Tsky>
    global interp_values
    return (gain * interp_values + offset)


def match_model_to_data(pol, fi, params):
    # Take model (params correspond to fit parameters from curve_to_fit)
    # and create curve that should match the observed data (ie. apply gains to theory)
    global Tsky_prime
    interp_values = Tsky_prime[pol, fi, :]
    return (params[0] * interp_values + params[1])


def match_data_to_model(data, params, mean):
    # Use model (params correspond to fit parameters from curve_to_fit)
    # to correct the data to match the theoretical Tsky (ie. apply gains to data)
    return (data - params[1]) / params[0] + mean

# Only run this script for HERA antennas
HERA_list = [  0,   1,   2,  11,  12,  13,  14,  23,  24,  25,  26,  27,  36,
        37,  38,  39,  40,  41,  50,  51,  52,  53,  54,  55,  65,  66,
        67,  68,  69,  70,  71,  82,  83,  84,  85,  86,  87,  88,  98,
       120, 121, 122, 123, 124, 136, 137, 138, 139, 140, 141, 142, 143]

# Separate by receiver
nrao_feeds = [51,65,66,70,82,83,98,120,121,122,123,124,141,143]
new_feeds = list(set(HERA_list)-set(nrao_feeds))

nant = len(HERA_list)
pols = ['X', 'Y']
npol = len(pols)
save_plots = True
save_data = False

# Load sky model from previous script:
# I ran the Tsky calc twice - once with Dave DeBoer's beam, and once with Nicolas
# Fagnoni's.
# Tsky_file = '/data4/tcox/HERA_IDR1_analysis/HERA_Tsky.npz'
Tsky_file = '../../HERA_Tsky_nic.npz'
data = np.load(Tsky_file)
freqs = data['freqs']  # These are already made to fit the data
model_lsts = data['lsts']
model_HERA_Tsky = data['HERA_Tsky']

# Load in data from previous script
autos_file = '../../IDR2_2458106_autos.npz'
data = np.load(autos_file)
data_lsts = data['lsts']
data_ave = data['data_ave']


'''
print data_lsts
data_lsts[data_lsts > 15] -= 24
idx = np.argsort(data_lsts, axis = 1)
data_lsts[0,:] = data_lsts[0,idx[0]]
data_lsts[1,:] = data_lsts[1,idx[1]]
data_ave = data_ave[:,:,idx[0],:]
'''

data_ave = data_ave[:,:,np.unique(np.where(data_lsts < 15)[1]),:]
lsts_new = data_lsts[(data_lsts < 15)]
data_lsts = lsts_new.reshape(2,int(data_lsts[(data_lsts < 15)].shape[0]/2))


# Fit auto data to model
auto_fits = np.zeros((npol, nant, len(freqs), 2))  # gain_amp, gain_slope, rxr_amp
covs = np.zeros((npol, nant, len(freqs), 2, 2))
# Create interp_func to return Tsky - <Tsky> (ie, deviation from mean)
interp_Tsky = interpolate.interp1d(model_lsts, model_HERA_Tsky, kind='cubic', axis=2,fill_value="extrapolate")
interp_Tsky_array = interp_Tsky(data_lsts[0])
Tsky_mean = np.mean(interp_Tsky_array, axis=2)
Tsky_prime = interp_Tsky_array - np.expand_dims(Tsky_mean, 2)
for pol in xrange(npol):
    for fi, freq in enumerate(freqs):
        interp_values = Tsky_prime[pol, fi, :]
        for anti, ant in enumerate(HERA_list):
            out = curve_fit(curve_to_fit, data_lsts[pol], data_ave[pol][ant, :, fi],
                            bounds=(0, np.inf))
            auto_fits[pol, anti, fi, :] = out[0]
            covs[pol, anti, fi, :, :] = out[1]

# Get gains and receiver temperatures
gains = auto_fits[:, :, :, 0]
rxr_temp = auto_fits[:, :, :, 1] / gains - Tsky_mean[:, None, :]

# rxr_temp is in K already. Need to convert gains to give Jy
# Jy = 2761.3006 * (T/K) * (m^2/Ae)
# beam_file = '/data4/tcox/instr_data/HERA_HFSS_X4Y2H_4900.hmap'
beam_file = '../../HERA_beam_nic.hmap'
beam = fits.getdata(beam_file, extname='BEAM_{0}'.format('X'))
beam_f = fits.getdata(beam_file, extname='FREQS_{0}'.format('X'))
func = interpolate.interp1d(beam_f, beam, kind='cubic', axis=1)
npix = beam.shape[0]
Ae = (aipy.const.c / (100.0 * freqs * 1.0e6))**2.0 / (4 * np.pi / npix * np.sum(func(freqs), axis=0))
K_to_Jy = 2761.3006 / Ae
gains_Jy = np.sqrt(gains / K_to_Jy.reshape(1, 1, -1))

outdir = '../2458106_new/'

if save_data:
    np.savez(outdir + 'fits.npz', gains = gains,
             rxr_temp = rxr_temp, HERA_list = HERA_list,
             nrao_feeds = nrao_feeds, freqs = freqs)

if save_plots:
    # first plot the actual fits

    # outdir = '/data4/tcox/HERA_IDR1_analysis/plots_new/'
    finds = [10, 32, 54]  # beginning, middle, end, but not at the very edges
    dark_colors = ['blue', 'green', 'red']
    light_colors = ['dodgerblue', 'lightgreen', 'salmon']
    fig = plt.figure('auto fits')

    r2_scores_pol = []

    for pol in xrange(npol):
	r2_scores_ant = []
        for anti, ant in enumerate(HERA_list):
            # Fits plot
            fig.clf()
            tittext = 'Antenna ' + str(ant) + pols[pol] + ' (HERA)'
	    for i, fi in enumerate(finds):
		data = match_data_to_model(data_ave[pol][ant, :, fi],
                                     auto_fits[pol, anti, fi, :],
                                     Tsky_mean[pol, fi])
                model = interp_Tsky_array[pol, fi, :]
		r2 = r2_score(data, model)
		label = str(freqs[fi]) + r' MHz - $R^2$ = ' + str(r2)[:5]
		plt.plot(data_lsts[pol], data, '.',
                         ms=5, color=dark_colors[i], label=label)
                plt.plot(data_lsts[pol], model,
                         color=light_colors[i])
            plt.ylim([0, 1.3 * np.max(interp_Tsky_array[pol, finds[0], :])])
            plt.xlabel('LST (Hours)')
            plt.ylabel('Tsky')
            plt.title(tittext)
            plt.legend()
            outfile = outdir + 'Tsky_v_LST_fit' + str(ant) + pols[pol] + '.png'
            plt.savefig(outfile)

	    fig.clf()
	    r2_fit_scores = []
	    for fi, freq in enumerate(freqs):
		data = match_data_to_model(data_ave[pol][ant, :, fi],
                                     auto_fits[pol, anti, fi, :],
                                     Tsky_mean[pol, fi])
                model = interp_Tsky_array[pol, fi, :]
		r2 = r2_score(data, model)

		r2_fit_scores.append(r2)
	    plt.plot(freqs, r2_fit_scores)
	    plt.xlabel('Frequency (MHz)')
            plt.ylabel(r'$R^2$ Score')
            plt.ylim([0.0, 1.1])
	    plt.title(tittext)
            outfile = outdir + 'r2_score' + str(ant) + pols[pol] + '.png'
            plt.savefig(outfile, dpi=200)

	    r2_scores_ant.append(r2_fit_scores)

            # Apply gains to model
            fig.clf()
            for i, fi in enumerate(finds):
                label = str(freqs[fi]) + 'MHz'
                plt.plot(data_lsts[pol], data_ave[pol][ant, :, fi], '.',
                         ms=5, color=dark_colors[i], label=label)
                plt.plot(data_lsts[pol], match_model_to_data(pol, fi, auto_fits[pol, anti, fi, :]),
                         color=light_colors[i])
            plt.ylim([0, 1.3 * np.max(data_ave)])
            plt.xlabel('LST (Hours)')
            plt.ylabel('Data and gains applied to model')
            plt.title(tittext)
            plt.legend()
            outfile = outdir + 'data_v_LST_fit' + str(ant) + pols[pol] + '.png'
            plt.savefig(outfile, dpi=200)
	r2_scores_pol.append(r2_scores_ant)

    np.savez(outdir + 'r2_scores.npz', r2_scores = r2_scores_pol)
    # Plots to show gain vs Antenna and freq
    plt.close(fig)
    fig, axarr = plt.subplots(1, npol, sharey=True, figsize=(10, 5))
    for pol in xrange(npol):
        for anti, ant in enumerate(HERA_list):
            plot = axarr[pol].plot(freqs, gains_Jy[pol, anti, :], label=str(ant))
        axarr[pol].set_xlabel('Freq (MHz)')
        axarr[pol].set_title(pols[pol])
    axarr[0].set_ylabel('Gain (Jy units)')
    axarr[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    outfile = outdir + 'bandpasses.png'
    plt.savefig(outfile, dpi=200)

    plt.close(fig)
    fig, axarr = plt.subplots(1, npol, sharey=True, figsize=(10, 5))
    for pol in xrange(npol):
        for anti, ant in enumerate(HERA_list):
            plot = axarr[pol].plot(freqs, rxr_temp[pol, anti, :], label=str(ant))
        axarr[pol].set_xlabel('Freq (MHz)')
        axarr[pol].set_title(pols[pol])
        axarr[pol].set_ylim([0, 1500])
    axarr[0].set_ylabel('Receiver Temp (K)')
    axarr[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    outfile = outdir + 'rxr_temp_per_ant.png'
    plt.savefig(outfile, dpi=200)

    plt.close(fig)
    fig, axarr = plt.subplots(npol, sharex=True)
    for pol in xrange(npol):
        im = axarr[pol].imshow(gains_Jy[pol, :, :], extent=(freqs.min(), freqs.max(), 0, nant),
                               aspect='auto', interpolation='none', origin='lower',
                               vmin=0, vmax=gains_Jy.max())
        axarr[pol].set_ylabel('Antenna index')
        axarr[pol].set_title(pols[pol])
        box = axarr[pol].get_position()
        axarr[pol].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    axarr[-1].set_xlabel('Frequency (MHz)')
    cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Gain (Jy units)')
    outfile = outdir + 'all_gains.png'
    plt.savefig(outfile, dpi=200)

    # Receiver temp
    plt.close(fig)
    fig, axarr = plt.subplots(npol, sharex=True)
    for pol in xrange(npol):
        im = axarr[pol].imshow(rxr_temp[pol, :, :], extent=(freqs.min(), freqs.max(), 0, nant),
                               aspect='auto', interpolation='none', origin='lower',
                               vmin=0, vmax=1500)
        axarr[pol].set_ylabel('Antenna index')
        axarr[pol].set_title(pols[pol])
        box = axarr[pol].get_position()
        axarr[pol].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    axarr[-1].set_xlabel('Frequency (MHz)')
    cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Receiver Temp (K)')
    outfile = outdir + 'all_rxr_temps.png'
    plt.savefig(outfile, dpi=200)
