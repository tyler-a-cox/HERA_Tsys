from calc_Tsys import TskySim, auto_data
import matplotlib.pyplot as plt
import numpy as np


hera_beam_file = '/home/shane/data/uv_beam_vivaldi.fits'
Tsky_file = '/data4/shane/data/HERA_Tsky_vivaldi.npz'
autos_file = 'post_power_drop_autos.uvh5'
data_dir = '/data4/shane/data/2458536/'

fig_dir = '/data4/tcox/HERA_Tsys_plots/2458536/'

Tsky_sim = TskySim(Tsky_file = Tsky_file, beam_file = hera_beam_file,
                   f_min=50,f_max=250)
Tsky_sim.build_model()
auto_fits = auto_data(data_dir=data_dir, filestart='zen.*',
                      fileend='*HH.uvh5', autos_file=autos_file,
                      f_min=50.,f_max=250.)
auto_fits.build_model(Tsky_sim)

auto_fits.fit_data()
'''

Plot of the receiver temperature vs. frequency

'''

plt.figure(figsize = (16,6))
plt.plot(auto_fits.Trxr[(0,'E')],label='antenna 0')
plt.ylim([0,3e6])
plt.yscale('symlog')

plt.legend(loc = 'best', framealpha = 1)

x_ticks = np.linspace(0,1509,num=10,dtype=int)
plt.xticks(x_ticks,(np.around(auto_fits.uv.freq_array[0,x_ticks]*1e-6)).astype(int))

plt.title('Receiver Temperature as a Function of Antenna/Frequency (XX Pol) Data: 2458536',size=14,verticalalignment='bottom')
plt.xlabel('Frequency (MHz)',size=14)
plt.ylabel('Temperature (K)',size=14)
plt.savefig(fig_dir + 'rxr_temp.png')

plt.clf()


'''

Plot of the gain and noise parameters

'''
fig, ax1 = plt.subplots()

fig.set_figheight(6)
fig.set_figwidth(14)

color = 'tab:red'
ax1.set_title('Spectrum of Noise and Gain Fitted Parameters (Antenna 0, XX Pol) Data: 2458536',size=14)
ax1.set_ylabel('Noise Parameter',size=14,color=color)
ax1.set_xlabel('Frequency (MHz)',size = 14)
ax1.plot(auto_fits.Trxr[(0,'E')]*auto_fits.gains[(0,'E')],label='antenna 0',color = color)
ax1.tick_params(axis='y')
ax1.set_yscale('symlog')

x_ticks = np.linspace(0,1509,num=10,dtype=int)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels((np.around(auto_fits.uv.freq_array[0,x_ticks]*1e-6)).astype(int))

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Gain Parameter',size = 14,color=color)
ax2.plot(auto_fits.gains[(0,'E')],label='antenna 0',color = color)
ax2.tick_params(axis='y')
ax2.set_yscale('symlog')

fig.tight_layout()
plt.savefig(fig_dir + 'noise_gain_plot.png')


'''
Data & Model vs LST
Plot data vs corrupted Tsky
'''

poli = 0
pol = auto_fits.pols[poli]

titles = ['Observed Dates (Linear Scale)',
          'Fitted Model (Linear Scale)',
          'Difference (Symlog Scale)']

for ant in auto_fits.ants:
    plt.clf()
    d = np.ma.masked_where(auto_fits.uv.get_flags((ant, ant, auto_fits.rev_pol_map[pol])),
                           auto_fits.uv.get_data((ant, ant, auto_fits.rev_pol_map[pol])))

    mdl_plot = ((auto_fits.Tsky[poli, :, :] - auto_fits.Tsky_mean[poli][:]) *
                auto_fits.fits[(ant, pol)][0][:] + auto_fits.fits[(ant, pol)][1][:])

    diff = d-mdl_plot

    fig, axes = plt.subplots(nrows=1, ncols=3)

    fig.set_figheight(8)
    fig.set_figwidth(18)

    data = [d,mdl_plot,diff]


    for i, ax in enumerate(axes.flat):
        if i != 2:
            im = ax.imshow(np.abs(data[i]), vmin=0, vmax=4e7, cmap='inferno')
        else:
            im = ax.imshow(np.abs(data[i]), norm = SymLogNorm(linthresh=1, vmin=0, vmax=1e7), cmap='inferno')
        ax.set_title('{}'.format(titles[i]))
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(fig_dir + 'data_model_diff_ant_{}.png'.format(ant))
