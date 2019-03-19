from calc_Tsys import TskySim, auto_data
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.ticker import LogLocator
import matplotlib.gridspec as gridspec


hera_beam_file = '/home/shane/data/uv_beam_vivaldi.fits'
Tsky_file = '/data4/shane/data/HERA_Tsky_vivaldi.npz'

#autos_file = 'lowband_autos.uvh5'
#data_dir = '/data4/shane/data/2458504/'
#fig_dir = '/data4/shane/data/HERA_Tsys/2458504/'


#autos_file = 'post_power_drop_autos.uvh5'
#data_dir = '/data4/shane/data/2458536/'
#fig_dir = '/data4/shane/data/HERA_Tsys/2458536/'


autos_file = '2458551_autos.uvh5'
data_dir = '/data4/shane/data/2458551/'
fig_dir = '/data4/shane/data/HERA_Tsys/2458551/'


def colorbar_plotter(fig,ax,im,label):
    
    pos = ax.get_position()
    cbarax = fig.add_axes([pos.x0 + pos.width+0.003, pos.y0, 0.005, pos.height])
    cbar = fig.colorbar(im,cax=cbarax)
    cbar.ax.tick_params(labelsize=8) 
    cbar.ax.set_ylabel(label, labelpad=6)
    
    return cbar


def Plot_Tsky_Vs_LST(data,chanlist,outputfile):

	Nchan = len(chanlist)
	
	fig, axes = plt.subplots(nrows=2, ncols=1)
	fig.subplots_adjust(left=0.20,top=0.90,right=0.8,bottom=0.3,wspace=0.2,hspace=0.01)

	gridspec.GridSpec(5,1)
	color = ['r','b','g','k']

	ax1 = plt.subplot2grid((5,1), (0,0), colspan=2, rowspan=3)

	ax1.set_ylabel(r'$[$K$]$')
	ax1.tick_params(axis='both',direction='in',which='both')
	plt.setp(ax1.get_xticklabels(), visible=False)	
	

	for i in range(Nchan):
	
    		ax1.plot(np.abs(data[0])[:,chanlist[i]],ls=':',c=color[i],lw=1.5,label=r'Data(chan# %d)' %chanlist[i])		
    		ax1.plot(np.abs(data[1])[:,chanlist[i]],ls='--',c=color[i],dashes=(6,3),lw=1.5,label=r'Model')			

	handles, labels = ax1.get_legend_handles_labels()
	ax1.legend(handles,labels, fontsize=6, loc='upper right',ncol=2)


	ax2 = plt.subplot2grid((5,1), (3,0), colspan=1, rowspan=2)

	ax2.set_xlabel(r'LST',labelpad=-1)
	ax2.set_ylabel(r'Relative error (D-M/M)')
	#ax2.set_ylim(-0.5,0.5)
	ax2.tick_params(axis='both',direction='in',which='both')
	

	for i in range(Nchan):
		rms= np.sqrt(np.mean((data[2][:,chanlist[i]]/np.abs(data[1][:,chanlist[i]]))**2))
		ax2.plot(data[2][:,chanlist[i]]/np.abs(data[1][:,chanlist[i]]),ls='-',c=color[i],lw=1.5,label=r'rms = %.2f' %rms)

	ax2.axhline(y=0.0,color='k',linestyle=':',linewidth=1.5)

	handles2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(handles2,labels2,fontsize=6,ncol=2,loc='upper left')

	fig.tight_layout()
	fig.savefig(fig_dir + outputfile,dpi=300, bbox_inches='tight')


def Plot_Tsky_avg(data,outputfile):
		
	rms = np.sqrt(np.mean((data[2]/np.abs(data[1]))**2,axis=0)) 	

	fig, ax = plt.subplots(nrows=1,ncols=1)
	fig.subplots_adjust(left=0.30,top=0.80,right=0.7,bottom=0.5,wspace=0.2,hspace=0.01)
	
	ax.set_xlabel(r'channels',labelpad=-1)	
	ax.set_ylabel(r'rms ((d - M)/M)')
	ax.tick_params(axis='both',direction='in',which='both')
	
	#ax.set_ylim(-0.25,1.5)
	ax.plot(rms,ls='-',c='b',lw=1.5)

	fig.tight_layout()
	fig.savefig(fig_dir + outputfile,dpi=300, bbox_inches='tight')


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

plt.title('Receiver Temperature as a Function of Antenna/Frequency (XX Pol)',size=14,verticalalignment='bottom')
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
ax1.set_title('Spectrum of Noise and Gain Fitted Parameters (Antenna 0, XX Pol)',size=14)
ax1.set_ylabel('Noise Parameter',size=14,color=color)
ax1.set_xlabel('Frequency (MHz)',size = 14)
ax1.plot((auto_fits.Trxr[(0,'E')]+auto_fits.Tsky_mean[0])*auto_fits.gains[(0,'E')],label='antenna 0',color = color)
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
Data & Model vs LST (2-D)
'''

poli = 0
pol = auto_fits.pols[poli]

titles = ['Observed Dates (Linear Scale)',
          'Fitted Model (Linear Scale)',
          'Difference (Symlog Scale)']

for ant in auto_fits.ants:
    plt.clf()
    d = auto_fits.data2Tsky((ant, pol))

    mdl_plot = auto_fits.Tsky[poli, :, :]

    d = np.abs(d)
    mdl_plot = np.abs(mdl_plot)

    diff = (d-mdl_plot)/np.abs(mdl_plot)

    data = [d,mdl_plot,diff]

    fig, axes = plt.subplots(nrows=1, ncols=3)
    
    fig.set_figheight(6)
    fig.set_figwidth(22)
    for i, ax in enumerate(axes.flat):
        if i != 2:
            im = ax.pcolormesh(data[i], norm=SymLogNorm(linthresh=1,vmin=10,vmax=10000),cmap='inferno')
            colorbar_plotter(fig,ax,im,'K')
            
        else:
            im = ax.pcolormesh(data[i],vmin=-.75,vmax=0.75, cmap='bwr')
            colorbar_plotter(fig,ax,im,'K')
            
    plt.savefig(fig_dir + 'data_model_diff_ant_{}.png'.format(ant))

    
'''
Data & Model vs LST (1-D)
'''

chans = [500, 900, 1300]

Plot_Tsky_Vs_LST(data,chans,"Tsky_Vs_LST.png")

Plot_Tsky_avg(data,"Tsky_Vs_Nu.png")