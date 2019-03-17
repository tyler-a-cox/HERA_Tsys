import numpy as np
import matplotlib.pyplot as plt
import linsolve
import uvtools
import pyuvdata
import glob
from astropy.io import fits
from pygsm import GSMObserver
from scipy import interpolate
import os
from hera_qm import xrfi
from hera_qm import UVFlag
from pyuvdata import UVData
from matplotlib.colors import SymLogNorm

class TskySim():
    """ Class to run Tsky Simulations"""
    def __init__(self, Tsky_file=None, beam_file=None, df=1.0, f_min=100.0, f_max=200.0, dlst=0.5,
                 pols=['E', 'N'], lat=-30.7224, lon=21.4278, elev=1100):
        """
        Initialize class
        Args:
            Tsky_file: Filename of npz file calculated from this class
            beam_file: Filename of fits file with beam
            df: channel width (MHz)
            f_min: minimum frequency (MHz)
            f_max: maximum frequency (MHz)
            dlst: LST separation (hours) Note, assume simulating 24 hours.
            pols: Polarizations
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            elev: Elevation (feet?)
        """
        self.Tsky_file = Tsky_file
        self.beam_file = beam_file
        self.df = df
        self.f_min = f_min
        self.f_max = f_max
        self.dlst = dlst
        self.pols = pols
        self.lat = lat
        self.lon = lon
        self.elev = elev

        if self.Tsky_file is None:
            self.Tsky = None
            self.update_options()
        else:
            self.read_Tsky()

    def update_options(self):
        self.freqs = np.arange(self.f_min, self.f_max + self.df / 2.0, self.df)
        self.hours = np.arange(0.0, 24.0, self.dlst)
        self.lsts = np.zeros_like(self.hours)

    def read_Tsky(self):
        d = np.load(self.Tsky_file)
        self.freqs = d['freqs']
        self.lsts = d['lsts']
        self.Tsky = d['Tsky']
        self.pols = d['pols']
        try:
            self.Ae = d['Ae']
        except KeyError:
            print('Warning: Effective area not read from file.')

    def write_Tsky(self):
        np.savez(self.Tsky_file, Tsky=self.Tsky, freqs=self.freqs, lsts=self.lsts, pols=self.pols, Ae=self.Ae)

    def calc_Tsky(self):
        """
        Calculate the Tsky sim based on beam file and the GSM
        """
        if self.beam_file is None:
            raise(ValueError('Beam file is not defined, cannot compute Tsky simulation.'))
        # Otherwise go ahead and calculate
        self.Tsky = np.zeros((len(self.pols), len(self.freqs), len(self.lsts)))

        # First set up beam
        uvb = UVBeam()
        uvb.read_beamfits(self.beam_file)
        beam = {}
        rot_pol = {}
        #temp_f = fits.getdata(self.beam_file, extname='FREQS')
        temp_f = uvb.freq_array[0]*1e-6
        for i, pol in enumerate(self.pols):
            '''try:
                im = fits.getdata(self.beam_file, extname='BEAM_{0}'.format(pol))
                rot_pol[pol] = False
            except KeyError:
                # My example file only has one pol, need to rotate it (later)
                im = fits.getdata(self.beam_file, extname='BEAM_{0}'.format(self.pols[1 - i]))
                rot_pol[pol] = True
            func = interpolate.interp1d(temp_f, im, kind='cubic', axis=1)
            beam[pol] = func(self.freqs)'''
            rot_pol[pol] = False
            im = uvb.data_array[0,0,i].transpose()
            func = interpolate.interp1d(temp_f, im, kind='cubic', axis=1)
            beam[pol] = func(self.freqs)


        # Get the effective area
        self.Ae = np.zeros((len(self.pols), len(self.freqs)))
        for poli, pol in enumerate(self.pols):
            self.Ae[poli] = ((const.c.to('m*MHz').value / self.freqs)**2. /
                             (4 * np.pi / im.shape[0] * np.sum(beam[pol], axis=0) /
                              np.max(beam[pol], axis=0)))

        # Set up observer
        ov = GSMObserver()
        ov.lon = str(self.lon)
        ov.lat = str(self.lat)
        ov.elev = self.elev

        fig = plt.figure('Tsky_calc')  # Never found a way to not open a figure...
        for poli, pol in enumerate(self.pols):
            for fi, freq in enumerate(self.freqs):
                if rot_pol[pol]:
                    pol_ang = 0.0
                else:
                    pol_ang = 90.0
                temp_beam = hp.orthview(beam[pol][:, fi], rot=[pol_ang, 90], fig=fig.number,
                                        xsize=400, return_projected_map=True, half_sky=True)
                temp_beam[np.isinf(temp_beam)] = np.nan
                for ti, t in enumerate(self.hours):
                    plt.clf()
                    dt = datetime(2013, 1, 1, np.int(t), np.int(60.0 * (t - np.floor(t))),
                                  np.int(60.0 * (60.0 * t - np.floor(t * 60.0))))
                    self.lsts[ti] = Time(dt).sidereal_time('apparent', self.lon).hour
                    ov.date = dt
                    ov.generate(freq)
                    d = ov.view(fig=fig.number)
                    sky = hp.orthview(d, fig=fig.number, xsize=400, return_projected_map=True, half_sky=True)
                    sky[np.isinf(sky)] = np.nan
                    self.Tsky[poli, fi, ti] = np.nanmean(temp_beam * sky) / np.nanmean(temp_beam)

        inds = np.argsort(self.lsts)
        self.lsts = self.lsts[inds]
        self.Tsky = self.Tsky[:, :, inds]

    def build_model(self, buffer=10):
        """
        Create interpolation model for calculating Tsky and Ae at any frequency/LST.

        Args:
            buffer: (int) size of wrapping buffer to avoid edge effects in LST interpolation.
        """
        # Pad to avoid edge effects
        temp_lsts = np.concatenate([self.lsts[-buffer:] - 24., self.lsts, self.lsts[:buffer] + 24.])
        temp_Tsky = np.concatenate([self.Tsky[:, :, -buffer:], self.Tsky, self.Tsky[:, :, :buffer]], axis=2)
        self.mdl = {}
        self.mdlAe = {}
        for poli, pol in enumerate(self.pols):
            self.mdl[pol] = interpolate.RectBivariateSpline(self.freqs, temp_lsts, temp_Tsky[poli, :, :])
            self.mdlAe[pol] = interpolate.interp1d(self.freqs, self.Ae[poli])

class auto_data():
    """ Class to hold auto correlation data """
    def __init__(self, data_dir='/data6/HERA/data/2458042/KM_uvR_files/', filestart='zen.*',
                 dpols=['xx', 'yy'], fileend='*.uvR', autos_file='IDR1_autos.uvR',f_min = 100.0, f_max = 200.0):
        self.data_dir = data_dir
        self.autos_file = autos_file
        self.dpols = dpols
        self.pol_map = {'xx': 'E', 'yy': 'N'}
        self.pols = np.array([self.pol_map[p] for p in self.dpols])
        self.rev_pol_map = {'E': 'xx', 'N': 'yy'}
        # Read in data
        self.filestart = filestart
        self.fileend = fileend
        self.read_data(f_min,f_max)
        self.update_freq_array(f_min,f_max)

    def read_data(self, f_min, f_max, force_read=False):
        self.uv = pyuvdata.UVData()
        if os.path.exists(self.data_dir + self.autos_file) and not force_read:
            self.uv.read(self.data_dir + self.autos_file)
            holder_freq_array_1 = self.uv.freq_array <= f_max*1e6
            holder_freq_array_2 = self.uv.freq_array >= f_min*1e6
            holder_freq_array = np.full(holder_freq_array_1.shape, True, dtype=bool)
            holder_freq_array[holder_freq_array_1==False] = False
            holder_freq_array[holder_freq_array_2==False] = False
            self.uv.read(self.data_dir + self.autos_file,freq_chans = holder_freq_array)

        else:
            file_lists = np.sort(glob.glob(self.data_dir + self.filestart + self.fileend))
            uv_temp = pyuvdata.UVData()
            for i in range(len(file_lists)):
                if i == 0:
                    self.uv.read([file_lists[i]])
                    self.uv.select(ant_str='auto')
                else:
                    uv_temp.read([file_lists[i]])
                    uv_temp.select(ant_str='auto')
                    self.uv += uv_temp
            self.uv.write_uvh5(self.data_dir + self.autos_file)
        # Get some useful parameters
        self.lsts, ind = np.unique(self.uv.lst_array, return_index=True)
        order = np.argsort(ind)
        self.lsts = 24 * self.lsts[order] / (2 * np.pi)
        self.wrap = np.argmax(self.lsts)
        self.freqs = self.uv.freq_array.flatten() * 1e-6
        self.ants = self.uv.get_ants()

    def update_freq_array(self,f_min,f_max):
        '''
        Remove frequencies outside of the range (MHz) given by user, regardless of
        frequencies in uv files.
        '''
        ind = 0
        while ind < len(self.freqs):
            if (self.freqs[ind] < f_min) or (self.freqs[ind] > f_max):
                self.freqs = np.delete(self.freqs,ind)
            else:
                ind += 1

    def build_model(self, sim):
        """
        Build model of Tsky that matches the data
        Args:
            sim: TskySim object with interpolation model already built
        """
        self.Tsky = np.zeros((len(self.pols), len(self.lsts), len(self.freqs)))
        self.Ae = np.zeros((len(self.pols), len(self.freqs)))
        self.Tsky_mean = np.zeros((len(self.pols), len(self.freqs)))
        for poli, pol in enumerate(self.pols):
            self.Tsky[poli, :, :] = np.concatenate([sim.mdl[pol](self.freqs, self.lsts[:self.wrap + 1]),
                                                    sim.mdl[pol](self.freqs, self.lsts[self.wrap + 1:])], axis=1).T
            self.Tsky_mean[poli, :] = self.Tsky[poli, :, :].mean(axis=0)
            self.Ae[poli, :] = sim.mdlAe[pol](self.freqs)

    def _fits2gTrxr(self, all_chans=True, ch=600):
        """ Quick function to get from linear solution to physical parameters."""
        if all_chans:
            Ae = self.Ae
        else:
            Ae = self.Ae[:, ch]
        for ant, pol in self.fits.keys():
            poli = np.where(self.pols == pol)[0][0]
            self.gains[(ant, pol)] = np.sqrt(Ae[poli] / 2761.3006 * self.fits[(ant, pol)][0])
            self.Trxr[(ant, pol)] = self.fits[(ant, pol)][1] / self.fits[(ant, pol)][0] - self.Tsky_mean[poli]

    def fit_data(self, all_chans=True, ch=600):
        """
        Fit gains and receiver temperatures based on LST evolution of signal fit to
        simulated Tsky.

        Args:
            all_chans: (bool) fit all channels if set to True (default, slow).
                       Otherwise only fit channel ch (faster).
            ch:        (int) Only fit this channel number. Default 600.
                       Ignored if all_chans == True.
        """
        self.gains = {}
        self.Trxr = {}
        self.fits = {}
        for poli, pol in enumerate(self.pols):
            for ant in self.ants:
                data = np.abs(self.uv.get_data((ant, ant, self.rev_pol_map[pol])))
                flags = self.uv.get_flags((ant, ant, self.rev_pol_map[pol]))
                d_ls = {}
                w_ls = {}
                kwargs = {}
                freq_low = np.where(self.uv.freq_array*1e-6 == np.min(self.freqs))[1][0]
                freq_high = np.where(self.uv.freq_array*1e-6 == np.max(self.freqs))[1][0]
                for i in range(self.lsts.size):
                    if all_chans:
                        # Solve for all channels at once
                        d_ls['Tsky%d*g+n' % i] = data[i, freq_low:(freq_high+1)]
                        w_ls['Tsky%d*g+n' % i] = 1 - flags[i, freq_low:(freq_high+1)]
                        kwargs['Tsky%d' % i] = self.Tsky[poli, i, :] - self.Tsky_mean[poli]
                    else:
                        # Only solve channel ch
                        d_ls['Tsky%d*g+n' % i] = data[i, ch]
                        w_ls['Tsky%d*g+n' % i] = 1 - flags[i, ch]
                        kwargs['Tsky%d' % i] = self.Tsky[poli, i, ch] - self.Tsky_mean[poli]
                ls = linsolve.LinearSolver(d_ls, w_ls, **kwargs)
                sol = ls.solve()
                self.fits[(ant, pol)] = (sol['g'], sol['n'])
        self._fits2gTrxr(all_chans=all_chans, ch=ch)

    def save_data(selffile_name = None )
        # Save data
        if save_file is not None:
            np.savez(file_name, fits = self.fits,
                                gains = self.gains,
                                Trxr = self.Trxr)

        elif save_file is None:
            print 'No save file given'

    def plot_data(self, ant, pol):
        pass

    def data2Tsky(self, key):
        poli = np.where(self.pols == key[1])[0]
        freq_low = np.where(self.uv.freq_array*1e-6 == np.min(self.freqs))[1][0]
        freq_high = np.where(self.uv.freq_array*1e-6 == np.max(self.freqs))[1][0]
        d = self.uv.get_data((key[0], key[0], self.rev_pol_map[key[1]]))[:,freq_low:(freq_high+1)]
        d = d / ((self.gains[key]**2. * 2761.3006 / self.Ae[poli]).reshape(1, -1)) - self.Trxr[key]
        return d

if __name__ == '__main__':
    hera_beam_file = '/home/shane/data/uv_beam_vivaldi.fits'
    Tsky_sim = TskySim(Tsky_file = '/data4/shane/data/HERA_Tsky_vivaldi.npz', beam_file = hera_beam_file,
                      f_min=50,f_max=250)
    Tsky_sim.build_model()
    auto_data = auto_data(data_dir='/data4/shane/data/2458536/', filestart='zen.*',
                             fileend='*HH.uvh5', autos_file='post_power_drop_autos.uvh5',f_min=50.,f_max=250.)
    auto_data.build_model(Tsky_sim)
    '''
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
        '''
