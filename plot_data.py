from calc_Tsys import TskySim, auto_data
import matplotlib.pyplot as plt

hera_beam_file = '/home/shane/data/uv_beam_vivaldi.fits'
Tsky_file = '/data4/shane/data/HERA_Tsky_vivaldi.npz'
autos_file = 'post_power_drop_autos.uvh5'
data_dir = '/data4/shane/data/2458536/'

Tsky_sim = TskySim(Tsky_file = Tsky_file, beam_file = hera_beam_file,
                   f_min=50,f_max=250)
Tsky_sim.build_model()
auto_data = auto_data(data_dir=data_dir, filestart='zen.*',
                      fileend='*HH.uvh5', autos_file=autos_file,
                      f_min=50.,f_max=250.)
auto_data.build_model(Tsky_sim)


if save_plots:
    # Rxr for one antenna
    plt.figure(figsize = (16,6))
    plt.plot(auto_data.Trxr[(0,'E')],label='antenna 0')
    plt.ylim([0,3e6])
    plt.yscale('symlog')

    plt.legend(loc = 'best', framealpha = 1)

    x_ticks = np.linspace(0,1509,num=10,dtype=int)
    plt.xticks(x_ticks,(np.around(auto_data.uv.freq_array[0,x_ticks]*1e-6)).astype(int))

    plt.title('Receiver Temperature as a Function of Antenna/Frequency (XX Pol) Data: 2458536',size=14,verticalalignment='bottom')
    plt.xlabel('Frequency (MHz)',size=14)
    plt.ylabel('Temperature (K)',size=14)
    plt.savefig('', dpi = 200)

    plt.clf()

    # Gain and Noise
    fig, ax1 = plt.subplots()

    fig.set_figheight(6)
    fig.set_figwidth(14)

    color = 'tab:red'
    ax1.set_title('Spectrum of Noise and Gain Fitted Parameters (Antenna 0, XX Pol) Data: 2458536',size=14)
    ax1.set_ylabel('Noise Parameter',size=14,color=color)
    ax1.set_xlabel('Frequency (MHz)',size = 14)
    ax1.plot(auto_data.Trxr[(0,'E')]*auto_data.gains[(0,'E')],label='antenna 0',color = color)
    ax1.tick_params(axis='y')
    ax1.set_yscale('symlog')

    x_ticks = np.linspace(0,1509,num=10,dtype=int)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels((np.around(auto_data.uv.freq_array[0,x_ticks]*1e-6)).astype(int))

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Gain Parameter',size = 14,color=color)
    ax2.plot(auto_data.gains[(0,'E')],label='antenna 0',color = color)
    ax2.tick_params(axis='y')
    ax2.set_yscale('symlog')

    fig.tight_layout()
    plt.savefig('', dpi = 200)

    plt.clf()

    # Data & Model vs LST
    plt.figure()
    chosen_ant = 1
    dark_colors = ['blue', 'green', 'red']
    light_colors = ['dodgerblue', 'lightgreen', 'salmon']
    poli = 0
    pol = auto_data.pols[poli]
    chans = [70,500,780] # channel size 820
    d = np.ma.masked_where(auto_data.uv.get_flags((chosen_ant, chosen_ant, auto_data.rev_pol_map[pol])),
                           auto_data.uv.get_data((chosen_ant, chosen_ant, auto_data.rev_pol_map[pol])))
    plot_lsts = np.concatenate([auto_data.lsts[:(auto_data.wrap+1)]-24, auto_data.lsts[(auto_data.wrap+1):]])
    for i, chan in enumerate(chans):
        d_plot = d[:, chan]
        plt.plot(plot_lsts, d_plot, '.', ms=1, color=dark_colors[i])
        mdl_plot = ((auto_data.Tsky[poli, :, chan] - auto_data.Tsky_mean[poli][chan]) *
                    auto_data.fits[(chosen_ant, pol)][0][chan] + auto_data.fits[(chosen_ant, pol)][1][chan])
        plt.plot(plot_lsts, mdl_plot, linewidth=0.5, color=light_colors[i],label=str(int(auto_data.freqs[chan]))+' MHz')
    plt.title('Raw data and fit')
    plt.legend()
