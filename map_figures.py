import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import argparse
import gdal
from scipy import stats
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import fmin_l_bfgs_b,minimize
import multiprocessing
import matplotlib.patches as mpatches
import subprocess

cl = [\
[120./255.,0./255.,0./255.],
[180./255.,0./255.,0./255.],
[250./255.,0./255.,0./255.],
[100./255.,100./255.,100./255.],
[0./255.,150./255.,255./255.],
[0./255.,50./255.,255./255.],
[0./255.,0./255.,180./255.]
]
rb_cm = matplotlib.colors.LinearSegmentedColormap.from_list('color_scheme',cl,N=256,gamma=1.0)


parser = argparse.ArgumentParser(description='Generate mapped figures')
parser.add_argument('-mask',default='cwc_dat_4km/v2017_year_2017_mask_match_cwd.tif')
parser.add_argument('-pers',default=0,type=int)
parser.add_argument('-cwc_thresh',default=0.1,type=float)
parser.add_argument('-pt_thresh',default=1.0,type=float)
args = parser.parse_args()

mask = gdal.Open(args.mask,gdal.GA_ReadOnly).ReadAsArray()
mask2 = (gdal.Open('cwc_dat_4km/2011_match_mask_cwd_mean.tif',gdal.GA_ReadOnly).ReadAsArray() > 0.5).astype(int)
mask[mask2!=1]=0
del mask2
cwc_preface = 'cwc_dat_4km/cwc_match_cwd_'

start_year_wye = 1990
end_year_wye = 2018


cwd_annual = []
prism_annual = []
tmax_annual = []
cwc_annual = []
titles = []
for year in range(start_year_wye,end_year_wye):

 # cwd
 dataset =  gdal.Open('cwd_dat_4km/california_cwd_' + str(year-1) + '.tif',gdal.GA_ReadOnly)
 annual = np.zeros((dataset.RasterYSize,dataset.RasterXSize))

 for b in range(8,13):
   ld = dataset.GetRasterBand(b).ReadAsArray()
   ld[ld == -9999] = np.nan
   ld[mask == 0] = np.nan
   annual += ld

 dataset =  gdal.Open('cwd_dat_4km/california_cwd_' + str(year) + '.tif',gdal.GA_ReadOnly)
 for b in range(1,8):
   ld = dataset.GetRasterBand(b).ReadAsArray()
   ld[ld == -9999] = np.nan
   ld[mask == 0] = np.nan
   annual += ld
 cwd_annual.append(annual)

 # prism precip
 dataset =  gdal.Open('precip_dat_4km/california_annual_precip_wye_' + str(year) + '.tif',gdal.GA_ReadOnly)
 ld = dataset.ReadAsArray()
 ld[ld == -9999] = np.nan
 ld[mask == 0] = np.nan
 prism_annual.append(ld)

# prism temp
 dataset =  gdal.Open('temp_dat_4km/california_annual_tmax_wye_' + str(year) + '.tif',gdal.GA_ReadOnly)
 ld = dataset.ReadAsArray()/12.
 ld[ld == -9999] = np.nan
 ld[mask == 0] = np.nan
 tmax_annual.append(ld)

 titles.append('WYE: ' + str(year))
 cwcf = cwc_preface + str(year) + '.tif'
 print(cwcf)
 # cwc
 if (os.path.isfile(cwcf)):
  dataset = gdal.Open(cwcf)
  ld = dataset.ReadAsArray()
  ld[ld == -9999] = np.nan
  ld[mask == 0] = np.nan
  cwc_annual.append(ld)
 else:
  ld = prism_annual[-1].copy()
  ld[:,:] = np.nan
  cwc_annual.append(ld)

cwc_delta = []
for n in range(1,len(cwc_annual)):
  cwc_delta.append((cwc_annual[n]-cwc_annual[n-1])/cwc_annual[n-1])


st = np.stack(prism_annual)
prism_mean = np.nanmedian(st,axis=0)
prism_sd = np.nanstd(st,axis=0)

st = np.stack(tmax_annual)
tmax_mean = np.nanmedian(st,axis=0)
tmax_sd = np.nanstd(st,axis=0)

st = np.stack(cwd_annual)
cwd_mean = np.nanmedian(st,axis=0)

st = np.stack(cwc_annual)
cwc_mean = np.nanmedian(st,axis=0)

cwc_annual[titles.index('WYE: 2012')][:,:] =  (cwc_annual[titles.index('WYE: 2011')][:,:] + cwc_annual[titles.index('WYE: 2013')][:,:])/2.


def crop_image(dat):

  x_has_nodata = np.any(np.isnan(dat) == False,axis=0).astype(int)
  y_has_nodata = np.any(np.isnan(dat) == False,axis=1).astype(int)

  trim_x_b = np.argmax(x_has_nodata)
  trim_x_t = np.argmax(x_has_nodata[::-1])
  trim_y_b = np.argmax(y_has_nodata)
  trim_y_t = np.argmax(y_has_nodata[::-1])
  
  dat = dat[trim_y_b:-trim_y_t,trim_x_b:-trim_x_t]
  return(dat)


mapped_upper_left_y = []
mapped_upper_left = []
mapped_md_length = []

cwc_frac = args.cwc_thresh
pers = bool(args.pers)

for n in range(3,len(cwc_annual)):
  mapped_md_length.append(cwc_annual[n].copy())
  mapped_md_length[-1][:,:] = np.nan

  lx = []
  ly = (cwc_annual[n].flatten()-cwc_mean.flatten())/cwc_mean.flatten()
  for m in range(n-3,n+1):
    ll = (prism_annual[m].flatten()-prism_mean.flatten())/prism_sd.flatten()
    lt = (tmax_annual[m].flatten()-tmax_mean.flatten())/tmax_sd.flatten()
    t = args.pt_thresh
    ll[lt >  t] = ll[lt >  t] * ( t+lt[lt >  t])**2
    lx.append(ll)

  lx = np.transpose(np.vstack(lx))
  lx[mask.flatten() == 0,:] = np.nan

  # get binary variables indicating if precip is positive or negative
  neg_x = np.logical_and(lx <= -args.pt_thresh,np.isnan(lx) == False)

  if (pers):
    valid = np.zeros(neg_x.shape).astype(bool)
    valid[:,-1] = neg_x[:,-1].copy()
    for m in range(1,4):
      valid[np.logical_and(valid[:,-m],neg_x[:,-(1+m)]) ,-(1+m)] = True
  else:
    valid = neg_x.copy()

  to_map = lx.copy()
  to_map[np.logical_not(valid)] = 0
  to_map = np.nansum(to_map,axis=1)
  to_map[ly <= -cwc_frac] = np.nan
  to_map[mask.flatten() == 0] = np.nan

  mapped_upper_left.append(to_map.reshape(cwc_annual[n].shape))

  to_map = valid.copy().astype(float)
  to_map = np.sum(to_map,axis=1)
  to_map[ly <= -cwc_frac] = np.nan
  to_map[mask.flatten() == 0] = np.nan

  mapped_upper_left_y.append(to_map.reshape(cwc_annual[n].shape))




####### Lag Calcs

lag_dat = np.zeros(mask.shape)
lag_dat[mask == 0] = np.nan
lag_dat[(cwc_annual[-1]-cwc_mean)/cwc_mean < -cwc_frac] = 1
lag_dat[np.logical_and((cwc_annual[-2]-cwc_mean)/cwc_mean < -cwc_frac,lag_dat==1)] = 2
lag_dat[np.logical_and((cwc_annual[-3]-cwc_mean)/cwc_mean < -cwc_frac,lag_dat==2)] = 3
lag_dat[np.logical_and((cwc_annual[-4]-cwc_mean)/cwc_mean < -cwc_frac,lag_dat==3)] = 4


########## Resistance Calcs #############################
st = np.stack(mapped_md_length)
drought_length = np.squeeze(np.nanmax(st,axis=0))
drought_length = crop_image(drought_length)

st = -1*np.stack(mapped_upper_left)
resistance_max = np.squeeze(np.nanmax(st,axis=0))
resistance_max = crop_image(resistance_max)

st = np.stack(mapped_upper_left_y)
resistance_max_y = np.squeeze(np.nanmax(st,axis=0))
resistance_max_y = crop_image(resistance_max_y)




####### Depending on settings, produces Figs 5,S11-14

fs = 13
fig = plt.figure(figsize=(10.,6.))
gs1 = gridspec.GridSpec(1,2)
gs1.update(wspace=0.0001,hspace=0.0001)

ax = plt.subplot(gs1[0])
im = plt.imshow(resistance_max,cmap='plasma',vmin=0,vmax = 20)
plt.axis('off')

if (pers):
  plt.text(48,0,'Magnitude of Consecutive\nDrought Resistance',ha='center',va='bottom',fontsize=fs)
else:
  plt.text(48,0,'Magnitude of\nDrought Resistance',ha='center',va='bottom',fontsize=fs)

gs1.update(right=0.8)
cbar_ax1 = fig.add_axes([0.36,0.57,0.02,0.25])
clb = plt.colorbar(im, cax=cbar_ax1,ticks=[0,5,10,15,20])
plt.title('Magnitude',fontsize=fs-1)


ax = plt.subplot(gs1[1])
im = plt.imshow(resistance_max_y,cmap='plasma',vmin=0,vmax=4)
plt.axis('off')
if (pers):
  plt.text(48,0,'Duration of Consecutive\nDrought Resistance',ha='center',va='bottom',fontsize=fs)
else:
  plt.text(48,0,'Duration of\nDrought Resistance',ha='center',va='bottom',fontsize=fs)

gs1.update(right=0.8)
cbar_ax = fig.add_axes([0.71,0.57,0.02,0.25])
clb = plt.colorbar(im, cax=cbar_ax,ticks=[0,1,2,3,4])
plt.title('Years',fontsize=fs-1)

if (pers):
  plt.savefig('figs/ir_resistance_pers_pt_' + str(args.pt_thresh) + '_cwc_' + str(args.cwc_thresh) + '.png',dpi=500,bbox_inches='tight')
else:
  plt.savefig('figs/ir_resistance_pt_' + str(args.pt_thresh) + '_cwc_' + str(args.cwc_thresh) + '.png',dpi=500,bbox_inches='tight')


 
################ Figure 4 - Lagged Drought Effect ##################
fig = plt.figure(figsize=(5.,6.))
gs1 = gridspec.GridSpec(1,1)
gs1.update(wspace=0.0001,hspace=0.0001)
ax = plt.subplot(gs1[0])
lag_dat = crop_image(lag_dat)
im = plt.imshow(lag_dat,cmap='plasma',vmin=0.0,vmax=4.0)
plt.axis('off')
plt.title('2017 Persistent CWC Loss',loc='left')
gs1.update(right=0.8)
cbar_ax = fig.add_axes([0.71,0.57,0.02,0.25])
clb = plt.colorbar(im, cax=cbar_ax,ticks=[0,1,2,3,4])
plt.title('Years',fontsize=fs-1)

plt.savefig('figs/figure_4.png',dpi=500,bbox_inches='tight')




