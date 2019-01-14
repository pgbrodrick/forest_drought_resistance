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


parser = argparse.ArgumentParser(description='Generate non-mapped figures')
parser.add_argument('-mask',default='cwc_dat_4km/v2017_year_2017_mask_match_cwd.tif')
parser.add_argument('-ecoregion',default=None,type=int)
parser.add_argument('-figure',default=0,type=int)
args = parser.parse_args()

mask = gdal.Open(args.mask,gdal.GA_ReadOnly).ReadAsArray()
mask2 = (gdal.Open('cwc_dat_4km/2011_match_mask_cwd_mean.tif',gdal.GA_ReadOnly).ReadAsArray() > 0.8).astype(int)
mask[mask2!=1]=0
del mask2

if args.ecoregion is not None:
  if (args.ecoregion < 1 or args.ecoregion > 10):
    Exception('Illegal Ecoregions - valid range 1-10')
  ecoregions = gdal.Open('cwc_dat_4km/2011_match_mask_calveg_ecoprovinces.tif',gdal.GA_ReadOnly).ReadAsArray().astype(float)
  mask[ecoregions != args.ecoregion] = 0

cwc_preface = 'cwc_dat_4km/cwc_match_cwd_'

start_year_wye = 1990
end_year_wye = 2018


cwd_annual = []
prism_annual = []
tmax_annual = []
vpd_annual = []
cwc_annual = []
pdsi_annual = []
titles = []
for year in range(start_year_wye,end_year_wye):

 # cwd
 dataset =  gdal.Open('cwd_dat_4km/california_cwd_' + str(year-1) + '.tif',gdal.GA_ReadOnly)
 annual = np.zeros((dataset.RasterYSize,dataset.RasterXSize))

 #for b in range(8,13):
 for b in range(11,13):
   ld = dataset.GetRasterBand(b).ReadAsArray()
   ld[ld == -9999] = np.nan
   ld[mask == 0] = np.nan
   annual += ld

 dataset =  gdal.Open('cwd_dat_4km/california_cwd_' + str(year) + '.tif',gdal.GA_ReadOnly)
 #for b in range(1,8):
 for b in range(1,4):
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

# vpd temp
 dataset =  gdal.Open('vpd_dat_4km/california_annual_vpd_wye_' + str(year) + '.tif',gdal.GA_ReadOnly)
 ld = dataset.ReadAsArray()/12.
 ld[ld == -9999] = np.nan
 ld[mask == 0] = np.nan
 vpd_annual.append(ld)

# pdsi
 dataset =  gdal.Open('pdsi_dat_4km/california_annual_pdsi_wye_' + str(year) + '.tif',gdal.GA_ReadOnly)
 ld = dataset.ReadAsArray()/12.
 ld[ld == -9999] = np.nan
 ld[mask == 0] = np.nan
 pdsi_annual.append(ld)







 titles.append('WYE: ' + str(year))
 cwcf = cwc_preface + str(year) + '.tif'
 print(cwcf)
 if (os.path.isfile(cwcf)):
  
  # cwc
  dataset = gdal.Open(cwcf)
  ld = dataset.ReadAsArray()
  ld[ld == -9999] = np.nan
  ld[mask == 0] = np.nan
  cwc_annual.append(ld)
 else:
  ld = prism_annual[-1].copy()
  ld[:,:] = np.nan
  cwc_annual.append(ld)


uq = 75
lq = 25
qs = 1.35
st = np.stack(prism_annual)
prism_mean = np.nanmedian(st,axis=0)
prism_sd = np.nanstd(st,axis=0)

st = np.stack(tmax_annual)
tmax_mean = np.nanmedian(st,axis=0)
tmax_sd = np.nanstd(st,axis=0)

st = np.stack(vpd_annual)
vpd_mean = np.nanmedian(st,axis=0)
vpd_sd = np.nanstd(st,axis=0)

st = np.stack(pdsi_annual)
pdsi_mean = np.nanmedian(st,axis=0)
pdsi_sd = np.nanstd(st,axis=0)

st = np.stack(cwd_annual)
cwd_mean = np.nanmedian(st,axis=0)
cwd_sd = np.nanstd(st,axis=0)

st = np.stack(cwc_annual)
cwc_mean = np.nanmedian(st,axis=0)
cwc_sd = np.nanstd(st,axis=0)

cwc_annual[titles.index('WYE: 2012')][:,:] =  (cwc_annual[titles.index('WYE: 2011')][:,:] + cwc_annual[titles.index('WYE: 2013')][:,:])/2.
cwc_delta = []
cwc_delta.append(cwc_annual[-1].copy())
cwc_delta[0][:,:] = np.nan
for n in range(1,len(cwc_annual)):
  cwc_delta.append((cwc_annual[n]-cwc_annual[n-1])/cwc_annual[n-1])


cwc_med_dev = [(x.flatten()-cwc_mean.flatten())/cwc_mean.flatten() for x in cwc_annual]
prism_med_dev = [(x.flatten()-prism_mean.flatten())/prism_sd.flatten() for x in prism_annual]
tmax_med_dev = [(x.flatten()-tmax_mean.flatten())/tmax_sd.flatten() for x in tmax_annual]
vpd_med_dev = [(x.flatten()-vpd_mean.flatten())/vpd_sd.flatten() for x in vpd_annual]
cwd_med_dev = [(cwd_mean.flatten()-x.flatten())/cwd_sd.flatten() for x in cwd_annual]
pdsi_med_dev = [(pdsi_mean.flatten()-x.flatten())/pdsi_sd.flatten() for x in pdsi_annual]

combo_annual = []
for n in range(len(prism_med_dev)):
  t = tmax_annual[n] - tmax_mean[n].copy()
  d = prism_annual[n] - prism_mean[n].copy()
  valid = t > 1
  d[valid] = d[valid] * ( 1+t[valid])**2
  combo_annual.append(d)

st = np.stack(combo_annual)
sd = np.nanstd(st,axis=0).flatten()

combo_mean = np.nanmedian(st,axis=0)
combo_sd = np.nanstd(st,axis=0)

years_drought_cwc = []
years_drought_prism = []
years_drought_tmax = []
years_drought_vpd = []
years_drought_cwd = []
years_drought_pdsi = []
years_drought_combo = []
annual_tmax = []

per_drought_cwc = []
per_drought_prism = []
per_drought_tmax = []
per_drought_vpd = []
per_drought_cwd = []
per_drought_pdsi = []
per_drought_combo = []

pws = []

offset_years = 4
threshold = 1
for n in range(offset_years,len(cwc_med_dev)):

   ######### PWS ##############
   lc = []
   for m in range(n-offset_years-1,n+1):
     lc.append(cwc_annual[m].flatten())
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   lpws = np.zeros(len(lc))
   for m in range(1,lc.shape[1]):
     loss = (lc[:,m-1] - lc[:,m])/lc[:,m-1]
     loss[np.logical_or.reduce((loss < 0,np.isnan(loss),np.isinf(loss)))] = 0
     lpws += loss
   pws.append(np.nanmean(lpws ))


   ############## CWC Anomaly ############### 
   lc = []
   for m in range(n-(offset_years-1),n+1):
     lc.append(cwc_med_dev[m].copy())
     #lc.append(cwc_delta[m].flatten())
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = lc < -0.10
   llc[np.isnan(lc)] = False
   years_drought_cwc.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   per_drought_cwc.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   ############### Prism Anomaly #################
   lc = []
   for m in range(n-(offset_years-1),n+1):
     lc.append(prism_med_dev[m].copy())
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = (lc < -threshold).astype(float)
   years_drought_prism.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   per_drought_prism.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   #################### Tmax Anomaly ###################
   lc = []
   for m in range(n-(offset_years-1),n+1):
     lc.append(tmax_med_dev[m].copy())
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = lc
   annual_tmax.append(np.nanmean( (np.nanmean(llc.astype(float) ,axis=1) ).astype(float) ))
   llc = (lc > threshold).astype(float)
   years_drought_tmax.append(np.nanmean( (np.nanmean(llc.astype(float) ,axis=1) ).astype(float) ))

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   per_drought_tmax.append(np.nanmean( (np.nanmean(llc.astype(float) ,axis=1) ).astype(float) ))

   #################### VPD Anomaly ###################
   lc = []
   for m in range(n-(offset_years-1),n+1):
     lc.append(vpd_med_dev[m].copy())
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = lc
   vpd_annual.append(np.nanmean( (np.nanmean(llc.astype(float) ,axis=1) ).astype(float) ))
   llc = (lc > threshold).astype(float)
   years_drought_vpd.append(np.nanmean( (np.nanmean(llc.astype(float) ,axis=1) ).astype(float) ))

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   per_drought_vpd.append(np.nanmean( (np.nanmean(llc.astype(float) ,axis=1) ).astype(float) ))

   ############### Combo Anomaly #################
   lc = []
   for m in range(n-(offset_years-1),n+1):
     d = prism_med_dev[m].copy()
     t = threshold
     d[tmax_med_dev[m] >  t] = d[tmax_med_dev[m] >  t] * ( t+tmax_med_dev[m][tmax_med_dev[m] >   t])**2
     lc.append(d)
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = (lc < -threshold).astype(float)
   years_drought_combo.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   per_drought_combo.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   ############### CWD Anomaly ################
   lc = []
   for m in range(n-(offset_years-1),n+1):
     lc.append(cwd_med_dev[m])
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = lc < -threshold

   years_drought_cwd.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   per_drought_cwd.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   ############### PDSI Anomaly ################
   lc = []
   for m in range(n-(offset_years-1),n+1):
     lc.append(pdsi_med_dev[m])
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = lc > threshold

   years_drought_pdsi.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   per_drought_pdsi.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))
 


state_cwc = [np.nanmean(x) for x in cwc_annual]
state_precip = [np.nanmean(x) for x in prism_annual]
state_cwd = [np.nanmean(x) for x in cwd_annual]
state_vpd = [np.nanmean(x) for x in vpd_annual]
state_pdsi = [np.nanmean(x) for x in pdsi_annual]
xt = [x[-2:] for x in titles]
xt.append('18')
xt = np.array(xt)[::2]
    






################ Figure 3 - Mean Deviation Years and Persistent Deviation Years ##################
if (args.figure == 3 or args.figure == 0):
  plt.figure(figsize=(14/2.,12/2.))
  gs1 = gridspec.GridSpec(2,2,width_ratios=[1,0.75])
  gs1.update(wspace=0.70,hspace=0.5)
  
  ax = plt.subplot(gs1[0,0])
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.ylabel('$\mathrm{CWC_{dur}}$ [years]')
  plt.xlabel('Year')
  plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6])
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_combo,c='green')
  ax2.set_ylabel('Mean Years of\nMeteorological Deviation')
  plt.ylabel('$\mathrm{PT_{dur}}$ [years]',{'color':'green'})
  
  
  
  ax = plt.subplot(gs1[1,0])
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.xlabel('Year')
  plt.yticks([0.0,0.1,0.2,0.3,0.4])
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_combo,c='green')
  plt.ylabel('$\mathrm{PT_{c.dur}}$ [years]',{'color':'green'})
  plt.yticks([0.0,0.5,1.0,1.5,2.0])
  
  
  ax = plt.subplot(gs1[0,1])
  plt.scatter(years_drought_combo,years_drought_cwc,c='green')
  slope, intercept, r_value, p_value, std_err = stats.linregress(years_drought_combo,years_drought_cwc)
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(years_drought_combo),np.max(years_drought_combo)],[np.min(years_drought_combo)*slope+intercept,np.max(years_drought_combo)*slope+intercept],ls='--',c='green')
  plt.text(np.min(years_drought_combo),np.max(years_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='top',ha='left')
  plt.xticks([0,0.5,1,1.5,2])
  plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6])
  
  plt.ylabel('$\mathrm{CWC_{dur}}$ [years]')
  plt.xlabel('$\mathrm{PT_{dur}}$ [years]')
  
  
  ax = plt.subplot(gs1[1,1])
  plt.scatter(per_drought_combo,per_drought_cwc,c='green')
  slope, intercept, r_value, p_value, std_err = stats.linregress(per_drought_combo,per_drought_cwc)
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(per_drought_combo),np.max(per_drought_combo)],[np.min(per_drought_combo)*slope+intercept,np.max(per_drought_combo)*slope+intercept],ls='--',c='green')
  plt.text(np.max(per_drought_combo),np.min(per_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='bottom',ha='right')
  plt.xticks([0,0.5,1,1.5,2])
  plt.yticks([0.0,0.1,0.2,0.3,0.4])
  
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.xlabel('$\mathrm{PT_{c.dur}}$ [years]')
  
  plt.savefig('figs/figure_3.png',dpi=200,bbox_inches='tight')




################ Figure S8 - Mean Deviation Years with scatter ##################
################ All
if (args.figure == -8 or args.figure == 0):
  plt.figure(figsize=(14/2.,36/2.))
  gs1 = gridspec.GridSpec(6,2,width_ratios=[1,0.75])
  gs1.update(wspace=0.70,hspace=0.5)
  
  
  ax = plt.subplot(gs1[0,0])
  
  plt.title('Precipitation')
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_prism,c='blue')
  ax2.set_ylabel('$\mathrm{Precip_{dur}}$ [years]',{'color':'blue'})
  
  ax = plt.subplot(gs1[0,1])
  plt.scatter(years_drought_prism,years_drought_cwc,c='blue')
  slope, intercept, r_value, p_value, std_err = stats.linregress(years_drought_prism,years_drought_cwc)
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(years_drought_prism),np.max(years_drought_prism)],[np.min(years_drought_prism)*slope+intercept,np.max(years_drought_prism)*slope+intercept],ls='--',c='blue')
  plt.text(np.min(years_drought_prism),np.max(years_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='top',ha='left')
  
  ax.set_xlabel('$\mathrm{Precip_{dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{dur}}$ [years]')
  
  
  
  
  ax = plt.subplot(gs1[1,0])
  
  plt.title('Climate Water Deficit')
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_cwd,c='grey')
  ax2.set_ylabel('$\mathrm{CWD_{dur}}$ [years]',{'color':'grey'})
  
  ax = plt.subplot(gs1[1,1])
  plt.scatter(years_drought_cwd,years_drought_cwc,c='grey')
  slope, intercept, r_value, p_value, std_err = stats.linregress(years_drought_cwd,years_drought_cwc)
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(years_drought_cwd),np.max(years_drought_cwd)],[np.min(years_drought_cwd)*slope+intercept,np.max(years_drought_cwd)*slope+intercept],ls='--',c='grey')
  plt.text(np.min(years_drought_cwd),np.max(years_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='top',ha='left')
  
  ax.set_xlabel('$\mathrm{CWD_{dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{dur}}$ [years]')
  
  
  
  ax = plt.subplot(gs1[2,0])
  
  plt.title('Temperature')
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_tmax,c='red')
  ax2.set_ylabel('$\mathrm{Temp_{dur}}$ [years]',{'color':'red'})
  
  ax = plt.subplot(gs1[2,1])
  plt.scatter(years_drought_tmax,years_drought_cwc,c='red')
  slope, intercept, r_value, p_value, std_err = stats.linregress(years_drought_tmax,years_drought_cwc)
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(years_drought_tmax),np.max(years_drought_tmax)],[np.min(years_drought_tmax)*slope+intercept,np.max(years_drought_tmax)*slope+intercept],ls='--',c='red')
  plt.text(np.min(years_drought_tmax),np.max(years_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='top',ha='left')
  
  ax.set_xlabel('$\mathrm{Temp_{dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{dur}}$ [years]')
  
  
  
  ax = plt.subplot(gs1[3,0])
  
  plt.title('Vapor Pressure Deficit')
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$CWC_{dur}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_vpd,c='purple')
  ax2.set_ylabel('$VPD_{dur}$ [years]',{'color':'purple'})
  
  ax = plt.subplot(gs1[3,1])
  plt.scatter(years_drought_vpd,years_drought_cwc,c='purple')
  slope, intercept, r_value, p_value, std_err = stats.linregress(years_drought_vpd,years_drought_cwc)
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(years_drought_vpd),np.max(years_drought_vpd)],[np.min(years_drought_vpd)*slope+intercept,np.max(years_drought_vpd)*slope+intercept],ls='--',c='purple')
  plt.text(np.min(years_drought_vpd),np.max(years_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='top',ha='left')
  
  ax.set_xlabel('$\mathrm{VPD_{dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{dur}}$ [years]')
  
  
  ax = plt.subplot(gs1[4,0])
  
  plt.title('PDSI')
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_pdsi,c='brown')
  ax2.set_ylabel('$\mathrm{PDSI_{dur}}$ [years]',{'color':'brown'})
  
  ax = plt.subplot(gs1[4,1])
  plt.scatter(years_drought_pdsi,years_drought_cwc,c='brown')
  slope, intercept, r_value, p_value, std_err = stats.linregress(years_drought_pdsi,years_drought_cwc)
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(years_drought_pdsi),np.max(years_drought_pdsi)],[np.min(years_drought_pdsi)*slope+intercept,np.max(years_drought_pdsi)*slope+intercept],ls='--',c='brown')
  plt.text(np.min(years_drought_pdsi),np.max(years_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='top',ha='left')
  
  ax.set_xlabel('$\mathrm{PDSI_{dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{dur}}$ [years]')
  
  
  
  
  
  
  ax = plt.subplot(gs1[5,0])
  
  plt.title('PT')
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],years_drought_combo,c='green')
  ax2.set_ylabel('$\mathrm{PT_{dur}}$ [years]',{'color':'green'})
  
  ax = plt.subplot(gs1[5,1])
  plt.scatter(years_drought_combo,years_drought_cwc,c='green')
  slope, intercept, r_value, p_value, std_err = stats.linregress(years_drought_combo,years_drought_cwc)
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(years_drought_combo),np.max(years_drought_combo)],[np.min(years_drought_combo)*slope+intercept,np.max(years_drought_combo)*slope+intercept],ls='--',c='green')
  plt.text(np.min(years_drought_combo),np.max(years_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='top',ha='left')
  
  ax.set_xlabel('$\mathrm{PT_{dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{dur}}$ [years]')
  
  
  
  if (args.ecoregion is not None):
    plt.savefig('figs/figure_S8_er' + str(int(args.ecoregion)) + '.png',dpi=200,bbox_inches='tight')
  else:
    plt.savefig('figs/figure_S8.png',dpi=200,bbox_inches='tight')


################ Figure S9 - Consecutive Deviation Years with scatter ##################
if (args.figure == -9 or args.figure == 0):

  plt.figure(figsize=(14/2.,36/2.))
  gs1 = gridspec.GridSpec(6,2,width_ratios=[1,0.75])
  gs1.update(wspace=0.70,hspace=0.5)
  
  
  ax = plt.subplot(gs1[0,0])
  
  plt.title('Precipitation')
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_prism,c='blue')
  ax2.set_ylabel('$\mathrm{Precip_{c.dur}}$ [years]',{'color':'blue'})
  
  ax = plt.subplot(gs1[0,1])
  plt.scatter(per_drought_prism[:-2],per_drought_cwc[:-2],c='blue')
  plt.scatter(per_drought_prism[-2:],per_drought_cwc[-2:],c='blue',marker='x')
  slope, intercept, r_value, p_value, std_err = stats.linregress(per_drought_prism[:-2],per_drought_cwc[:-2])
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(per_drought_prism),np.max(per_drought_prism)],[np.min(per_drought_prism)*slope+intercept,np.max(per_drought_prism)*slope+intercept],ls='--',c='blue')
  plt.text(np.max(per_drought_prism),-0.02,'R$^2$='+str(round(r_value**2,2)),va='bottom',ha='right')
  
  ax.set_xlabel('$\mathrm{Precip_{c.dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  
  
  ax = plt.subplot(gs1[1,0])
  
  plt.title('Climate Water Deficit')
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_cwd,c='grey')
  ax2.set_ylabel('$\mathrm{CWD_{c.dur}}$ [years]',{'color':'grey'})
  
  ax = plt.subplot(gs1[1,1])
  plt.scatter(per_drought_cwd[:-2],per_drought_cwc[:-2],c='grey')
  plt.scatter(per_drought_cwd[-2:],per_drought_cwc[-2:],c='grey',marker='x')
  slope, intercept, r_value, p_value, std_err = stats.linregress(per_drought_cwd[:-2],per_drought_cwc[:-2])
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(per_drought_cwd),np.max(per_drought_cwd)],[np.min(per_drought_cwd)*slope+intercept,np.max(per_drought_cwd)*slope+intercept],ls='--',c='grey')
  plt.text(np.max(per_drought_cwd),-0.02,'R$^2$='+str(round(r_value**2,2)),va='bottom',ha='right')
  
  ax.set_xlabel('$\mathrm{CWD_{c.dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  
  
  
  ax = plt.subplot(gs1[2,0])
  
  plt.title('Temperature')
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_tmax,c='red')
  ax2.set_ylabel('$\mathrm{Temp_{c.dur}}$ [years]',{'color':'red'})
  
  ax = plt.subplot(gs1[2,1])
  plt.scatter(per_drought_tmax[:-2],per_drought_cwc[:-2],c='red')
  plt.scatter(per_drought_tmax[-2:],per_drought_cwc[-2:],c='red',marker='x')
  slope, intercept, r_value, p_value, std_err = stats.linregress(per_drought_tmax[:-2],per_drought_cwc[:-2])
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(per_drought_tmax),np.max(per_drought_tmax)],[np.min(per_drought_tmax)*slope+intercept,np.max(per_drought_tmax)*slope+intercept],ls='--',c='red')
  plt.text(np.max(per_drought_tmax),-0.02,'R$^2$='+str(round(r_value**2,2)),va='bottom',ha='right')
  
  ax.set_xlabel('$\mathrm{Temp{c.dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  
  
  
  ax = plt.subplot(gs1[3,0])
  
  plt.title('Vapor Pressure Deficit')
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_vpd,c='purple')
  ax2.set_ylabel('$\mathrm{VPD_{c.dur}}$ [years]',{'color':'purple'})
  
  ax = plt.subplot(gs1[3,1])
  plt.scatter(per_drought_vpd[:-2],per_drought_cwc[:-2],c='purple')
  plt.scatter(per_drought_vpd[-2:],per_drought_cwc[-2:],c='purple',marker='x')
  slope, intercept, r_value, p_value, std_err = stats.linregress(per_drought_vpd[:-2],per_drought_cwc[:-2])
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(per_drought_vpd),np.max(per_drought_vpd)],[np.min(per_drought_vpd)*slope+intercept,np.max(per_drought_vpd)*slope+intercept],ls='--',c='purple')
  plt.text(np.max(per_drought_vpd),-0.02,'R$^2$='+str(round(r_value**2,2)),va='bottom',ha='right')
  
  ax.set_xlabel('$\mathrm{VPD_{c.dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  
  
  ax = plt.subplot(gs1[4,0])
  
  plt.title('PDSI')
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_pdsi,c='brown')
  ax2.set_ylabel('$\mathrm{PDSI_{c.dur}}$ [years]',{'color':'brown'})
  
  ax = plt.subplot(gs1[4,1])
  plt.scatter(per_drought_pdsi[:-2],per_drought_cwc[:-2],c='brown')
  plt.scatter(per_drought_pdsi[-2:],per_drought_cwc[-2:],c='brown',marker='x')
  slope, intercept, r_value, p_value, std_err = stats.linregress(per_drought_pdsi[:-2],per_drought_cwc[:-2])
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(per_drought_pdsi),np.max(per_drought_pdsi)],[np.min(per_drought_pdsi)*slope+intercept,np.max(per_drought_pdsi)*slope+intercept],ls='--',c='brown')
  plt.text(np.max(per_drought_pdsi),-0.02,'R$^2$='+str(round(r_value**2,2)),va='bottom',ha='right')
  
  ax.set_xlabel('$\mathrm{PDSI_{c.dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  
  
  
  
  
  
  
  
  ax = plt.subplot(gs1[5,0])
  
  plt.title('PT')
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_cwc,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.xlabel('Year')
  
  ax2 = ax.twinx()
  plt.plot(np.arange(len(state_cwc))[offset_years:],per_drought_combo,c='green')
  ax2.set_ylabel('$\mathrm{PT_{c.dur}}$ [years]',{'color':'green'})
  
  ax = plt.subplot(gs1[5,1])
  plt.scatter(per_drought_combo[:-2],per_drought_cwc[:-2],c='green')
  plt.scatter(per_drought_combo[-2:],per_drought_cwc[-2:],c='green',marker='x')
  slope, intercept, r_value, p_value, std_err = stats.linregress(per_drought_combo[:-2],per_drought_cwc[:-2])
  print(('r2:',r_value**2))
  print(('p:',p_value))
  plt.plot([np.min(per_drought_combo),np.max(per_drought_combo)],[np.min(per_drought_combo)*slope+intercept,np.max(per_drought_combo)*slope+intercept],ls='--',c='green')
  plt.text(np.max(per_drought_combo),np.min(per_drought_cwc),'R$^2$='+str(round(r_value**2,2)),va='bottom',ha='right')
  
  ax.set_xlabel('$\mathrm{PT_{c.dur}}$ [years]')
  ax.set_ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  
  
  
  plt.savefig('figs/figure_S9.png',dpi=200,bbox_inches='tight')






################ Figure 2  ##################
if (args.figure == 2 or args.figure == 0):
  plt.figure(figsize=(14/2.,18/2.))
  cwc = [np.nanmean(x) for x in cwc_annual]
  cwd = [np.nanmean(x) for x in cwd_annual]
  prism = [np.nanmean(x) for x in prism_annual]
  tmax = [np.nanmean(x) for x in tmax_annual]
  combo = [np.nanmean(x) for x in combo_annual]
  
  gs1 = gridspec.GridSpec(3,2,width_ratios=[1,0.75])
  gs1.update(wspace=0.70,hspace=0.5)
  
  ax = plt.subplot(gs1[0,0])
  plt.plot(np.arange(len(cwc)),np.array(cwc)/1000.,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.xlabel('Year')
  plt.ylabel('CWC [L m$^{-2}$]')
  ax2 = ax.twinx()
  plt.plot(np.arange(len(cwc)),prism,c='blue')
  plt.ylabel('Total Precipitation [mm]',{'color':'blue'})
  plt.title('Precipitation')
  
  
  ax = plt.subplot(gs1[1,0])
  plt.plot(np.arange(len(cwc)),np.array(cwc)/1000.,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.xlabel('Year')
  plt.ylabel('CWC [L m$^{-2}$]')
  ax2 = ax.twinx()
  plt.plot(np.arange(len(cwc)),-np.array(cwd),c='grey')
  plt.ylabel('Total Negative CWD [mm]',{'color':'grey'})
  plt.title('CWD')
  
  
  ax = plt.subplot(gs1[2,0])
  plt.plot(np.arange(len(cwc)),np.array(cwc)/1000.,c='black')
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.xlabel('Year')
  plt.ylabel('CWC [L m$^{-2}$]')
  ax2 = ax.twinx()
  plt.plot(np.arange(len(cwc)),-np.array(tmax),c='red')
  plt.ylabel('Mean Negative T-Max [C]',{'color':'red'})
  plt.title('Temperature')
  
  
  prism = np.array(prism) - np.mean(prism)
  prism = prism / np.std(prism)
  
  cwc = np.array(cwc) - np.mean(cwc)
  cwc = cwc / np.std(cwc)
  
  cwd = np.array(cwd) - np.mean(cwd)
  cwd = cwd / np.std(cwd)
  
  tmax = np.array(tmax) - np.mean(tmax)
  tmax = tmax / np.std(tmax)
  
  ax = plt.subplot(gs1[0,1])
  plt.scatter(prism,cwc,c='blue')
  slope, intercept, r_value, p_value, std_err = stats.linregress(prism,cwc)
  print(('p:',p_value))
  plt.plot([np.min(prism),np.max(prism)],[np.min(prism)*slope + intercept,np.max(prism)*slope + intercept],ls='--',c='blue')
  plt.ylabel('Normalized CWC')
  plt.xlabel('Normalized Precipitation')
  plt.xlim([-2.5,2.5])
  plt.ylim([-2.5,2.5])
  plt.xticks(np.arange(-2,3))
  plt.text(2.3,-2.3,'R$^2$='+str(round(r_value**2,2)),ha='right',va='bottom')
  
  ax = plt.subplot(gs1[1,1])
  plt.scatter(cwd,cwc,c='grey')
  slope, intercept, r_value, p_value, std_err = stats.linregress(cwd,cwc)
  print(('p:',p_value))
  plt.plot([np.min(cwd),np.max(cwd)],[np.min(cwd)*slope + intercept,np.max(cwd)*slope + intercept],ls='--',c='grey')
  plt.ylabel('Normalized CWC')
  plt.xlabel('Normalized CWD')
  plt.xlim([-2.5,2.5])
  plt.ylim([-2.5,2.5])
  plt.xticks(np.arange(-2,3))
  plt.text(2.3,2.3,'R$^2$='+str(round(r_value**2,2)),ha='right',va='top')
  
  ax = plt.subplot(gs1[2,1])
  plt.scatter(tmax,cwc,c='red')
  slope, intercept, r_value, p_value, std_err = stats.linregress(tmax,cwc)
  print(('p:',p_value))
  plt.plot([np.min(tmax),np.max(tmax)],[np.min(tmax)*slope + intercept,np.max(tmax)*slope + intercept],ls='--',c='red')
  plt.ylabel('Normalized CWC')
  plt.ylabel('Normalized Temperature')
  plt.xlim([-2.5,2.5])
  plt.ylim([-2.5,2.5])
  plt.xticks(np.arange(-2,3))
  plt.text(2.3,2.3,'R$^2$='+str(round(r_value**2,2)),ha='right',va='top')
  
  le = []
  le.append('Precipitation,  R$^2$='+str(round(r_value**2,2)))
  le.append('CWD,              R$^2$='+str(round(r_value**2,2)))
  le.append('Temperature, R$^2$='+str(round(r_value**2,2)))
  
  plt.ylabel('Normalized CWC')
  plt.xlabel('Normalized Temperature')
  
  
  if (args.ecoregion is not None):
    plt.savefig('figs/figure_2_er' + str(int(args.ecoregion)) + '.png',dpi=200,bbox_inches='tight')
  else:
    plt.savefig('figs/figure_2.png',dpi=200,bbox_inches='tight')






############### Sensitivity


############## CWC Anomaly ############### 
offset_years = 4
thresh_range_cwc = [0,-0.05,-0.075,-0.1,-0.15,-0.2]
sens_ydc = np.zeros((len(range(offset_years,len(cwc_med_dev))),len(thresh_range_cwc)))
sens_pdc = np.zeros((len(range(offset_years,len(cwc_med_dev))),len(thresh_range_cwc)))
for n in range(offset_years,len(cwc_med_dev)):
 for _t in range(len(thresh_range_cwc)):
   lc = []
   for m in range(n-(offset_years-1),n+1):
     lc.append(cwc_med_dev[m].copy())
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = lc < thresh_range_cwc[_t]
   llc[np.isnan(lc)] = False
   sens_ydc[n-offset_years,_t] = np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) )

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   per_drought_cwc.append(np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) ))
   sens_pdc[n-offset_years,_t] = np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) )


offset_years = 4
thresh_range_pt = [-0,-0.5,-0.75,-1.0,-1.5,-2]
sens_ydpt = np.zeros((len(range(offset_years,len(cwc_med_dev))),len(thresh_range_pt)))
sens_pdpt = np.zeros((len(range(offset_years,len(cwc_med_dev))),len(thresh_range_pt)))

############### Combo Anomaly #################
for n in range(offset_years,len(cwc_med_dev)):
 for _t in range(len(thresh_range_pt)):
   lc = []
   for m in range(n-(offset_years-1),n+1):
     d = prism_med_dev[m].copy()
     t = -thresh_range_pt[_t]
     d[tmax_med_dev[m] >  t] = d[tmax_med_dev[m] >  t] * ( t+tmax_med_dev[m][tmax_med_dev[m] >   t])**2
     lc.append(d)
   lc = np.transpose(np.vstack(lc))
   lc = lc[np.all(np.isnan(lc),axis=1) == False,:]
   llc = (lc < -t).astype(float)
   sens_ydpt[n-offset_years,_t] = np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) )

   for m in range((offset_years-2),-1,-1):
     llc[llc[:,m+1] ==0,m] = 0
   sens_pdpt[n-offset_years,_t] = np.nanmean( (np.nansum(llc.astype(float) ,axis=1) ).astype(float) )




################ Figure S10  ##################
if (args.figure == -10 or args.figure == 0):
  plt.figure(figsize=(14/2.,14/2.))
  gs1 = gridspec.GridSpec(2,2,width_ratios=[1,1])
  gs1.update(wspace=0.30,hspace=0.3)
  
  
  ax = plt.subplot(gs1[0,0])
  for _t in range(len(thresh_range_cwc)):
    plt.plot(np.arange(len(state_cwc))[offset_years:],sens_ydc[:,_t])
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.xlim(4,len(state_cwc))
  plt.xlabel('Years')
  plt.ylabel('$\mathrm{CWC_{dur}}$ [years]')
  plt.legend(thresh_range_cwc,loc='upper left')
  plt.yticks([0,0.5,1,1.5,2,2.5])
  
  ax = plt.subplot(gs1[0,1])
  for _t in range(len(thresh_range_cwc)):
    plt.plot(np.arange(len(state_cwc))[offset_years:],sens_pdc[:,_t])
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.xlim(4,len(state_cwc))
  plt.xlabel('Years')
  plt.ylabel('$\mathrm{CWC_{c.dur}}$ [years]')
  plt.legend(thresh_range_cwc,loc='upper left')
  plt.yticks([0,0.5,1,1.5,2])
  
  ax = plt.subplot(gs1[1,0])
  for _t in range(len(thresh_range_pt)):
    plt.plot(np.arange(len(state_cwc))[offset_years:],sens_ydpt[:,_t])
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.xlim(4,len(state_cwc))
  plt.xlabel('Years')
  plt.ylabel('$\mathrm{PT_{dur}}$ [years]')
  plt.legend(thresh_range_pt,loc='upper left')
  plt.yticks([0,1,2,3,4])
  
  ax = plt.subplot(gs1[1,1])
  for _t in range(len(thresh_range_pt)):
    plt.plot(np.arange(len(state_cwc))[offset_years:],sens_pdpt[:,_t])
  plt.xticks(np.arange(len(state_cwc)+1,step=2),xt,rotation=90)
  plt.xlim(4,len(state_cwc))
  plt.xlabel('Years')
  plt.ylabel('$\mathrm{PT_{c.dur}}$ [years]')
  plt.legend(thresh_range_pt,loc='upper left')
  plt.yticks([0,1,2,3,4])
  
  
  plt.savefig('figs/figure_S10.png',dpi=200,bbox_inches='tight')



