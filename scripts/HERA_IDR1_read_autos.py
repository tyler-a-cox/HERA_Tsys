import numpy as np
import capo
import aipy
from glob import glob

nant = 128
chanave = 16

xxglob = '/data6/HERA/data/2457458/*xx.uv'
yyglob = '/data6/HERA/data/2457458/*yy.uv'

xxfilenames = glob(xxglob)
yyfilenames = glob(yyglob)

xxt, xxd, xxf = capo.arp.get_dict_of_uv_data(xxfilenames, polstr='xx', antstr='auto')
yyt, yyd, yyf = capo.arp.get_dict_of_uv_data(yyfilenames, polstr='yy', antstr='auto')
xlsts = xxt['lsts'] * 12.0 / np.pi
ylsts = yyt['lsts'] * 12.0 / np.pi
xnt, nchan = xxd[0, 0]['xx'].shape
ynt = yyd[0, 0]['yy'].shape[0]
# Do some coarse averaging
xxd_ave = np.zeros((nant, xnt, nchan / chanave), dtype=np.float64)
yyd_ave = np.zeros((nant, ynt, nchan / chanave), dtype=np.float64)
for ant in xrange(128):
    for chan in xrange(nchan / chanave):
        xxd_ave[ant, :, chan] = np.real(np.mean(xxd[ant, ant]['xx'][:, (chan * chanave):((chan + 1) * chanave)], axis=1))
        yyd_ave[ant, :, chan] = np.real(np.mean(yyd[ant, ant]['yy'][:, (chan * chanave):((chan + 1) * chanave)], axis=1))

inds = np.argsort(xlsts)
xlsts = xlsts[inds]
xxd_ave = xxd_ave[:, inds, :]

inds = np.argsort(ylsts)
ylsts = ylsts[inds]
yyd_ave = yyd_ave[:, inds, :]

lsts = [xlsts, ylsts]
data_ave = [xxd_ave, yyd_ave]

autos_file = '/data4/tcox/HERA_IDR1_analysis/IDR1_autos.npz'
np.savez(autos_file, data_ave=data_ave, lsts=lsts)
