#### Created by: Philip Brodrick
#### Purpose: Munge CWC data for training
####          

import gdal
from gdalconst import *
import numpy as np
import sys
import subprocess
import os
import signal
from scipy import stats
import subprocess
import pandas as pd
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import cross_validation, metrics
import numpy.matlib
import time
import timeit
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_conv_dat(read_dat,ws,single_match=False):
 if (single_match == False):
  dat = []
  for mmm in range(0,ws):
   for nnn in range(0,ws):
     dat.append(read_dat[mmm:read_dat.shape[0]-ws+mmm,nnn:read_dat.shape[1]-ws+nnn].flatten())
  return dat
 else:
  mmm = int(round(ws/2.)-1)  
  nnn = int(round(ws/2.)-1)  
  return read_dat[mmm:read_dat.shape[0]-ws+mmm,nnn:read_dat.shape[1]-ws+nnn].flatten()

def get_conv_dat_masked(read_dat,ws,mask,single_match=False):
 if (single_match == False):
  dat = []
  for mmm in range(0,ws):
   for nnn in range(0,ws):
     dat.append(read_dat[mmm:read_dat.shape[0]-ws+mmm,nnn:read_dat.shape[1]-ws+nnn].flatten()[mask])
  return dat
 else:
  mmm = int(round(ws/2.)-1)  
  nnn = int(round(ws/2.)-1)  
  return read_dat[mmm:read_dat.shape[0]-ws+mmm,nnn:read_dat.shape[1]-ws+nnn].flatten()[mask]




year = sys.argv[1]
ws = 3
version = 'thresh10_ms'

filenames = []

filenames.append('dat/stack_me/x.tif')
filenames.append('dat/stack_me/y.tif')
filenames.append('dat/stack_me/elevation.tif')
filenames.append('dat/stack_me/slope.tif')
filenames.append('dat/stack_me/aspect.tif')
filenames.append('dat/stack_me/rem.tif')
filenames.append('dat/stack_me/road_distance.tif')
filenames.append('dat/stack_me/road_density.tif')
filenames.append('dat/stack_me/insol_sep_21.tif')
filenames.append('dat/stack_me/insol_mar_21.tif')
filenames.append('dat/stack_me/insol_jun_21.tif')
filenames.append('dat/stack_me/insol_dec_21.tif')

filenames.append('dat/2017_refl/proc/california_sr_' + year + '_7.dat')

filenames.append('dat/stack_me_300_30/elevation.tif')
filenames.append('dat/stack_me_300_30/slope.tif')
filenames.append('dat/stack_me_300_30/aspect.tif')
filenames.append('dat/stack_me_300_30/rem.tif')
filenames.append('dat/stack_me_300_30/road_distance.tif')
filenames.append('dat/stack_me_300_30/road_density.tif')
filenames.append('dat/stack_me_300_30/insol_sep_21.tif')
filenames.append('dat/stack_me_300_30/insol_mar_21.tif')
filenames.append('dat/stack_me_300_30/insol_jun_21.tif')
filenames.append('dat/stack_me_300_30/insol_dec_21.tif')
filenames.append('dat/2017_refl/proc_300_30/california_sr_' + year + '_7.tif')




col_i = 0
print('loading data....')
full_dat = []
dataset = gdal.Open('dat/y_dat/ca_' + year + '_cwc_30m_me.tif',gdal.GA_ReadOnly)
y_dat = dataset.GetRasterBand(1).ReadAsArray()
y_dat = get_conv_dat(y_dat,ws,True)

dataset = gdal.Open('dat/y_dat/ca_' + year + '_propvalid_30m_me.tif',gdal.GA_ReadOnly)
y_cover = dataset.GetRasterBand(1).ReadAsArray()
y_cover = get_conv_dat(y_cover,ws,True)


min_y = 0
max_y = 10000
good_dat = (y_dat > min_y)
good_dat[y_dat > max_y] = False
good_dat[y_cover < .10] = False


y_dat = y_dat[good_dat]

n_col = 0
for fi in filenames:
 dataset = gdal.Open(fi,gdal.GA_ReadOnly)
 if ('/x.tif' in fi or '/y.tif' in fi):
  band_dat = dataset.GetRasterBand(1).ReadAsArray()
  band_dat = get_conv_dat_masked(band_dat,ws,good_dat,True)
  full_dat.append(band_dat)
  print(('true',fi,1))
 else:
  for b in range(1,dataset.RasterCount+1):
   band_dat = dataset.GetRasterBand(b).ReadAsArray()
   band_dat = get_conv_dat_masked(band_dat,ws,good_dat)
   for _i in range(0,len(band_dat)):
     if ('aspect' in fi):
      full_dat.append(np.sin(band_dat[_i]*np.pi/180.))
     else:
      full_dat.append(band_dat[_i])
   print((fi,b,dataset.RasterCount))
print('loaded.  Masking...')
good_dat = np.ones(full_dat[0].shape).astype(bool)
for n in range(0,len(full_dat)):
  good_dat[full_dat[n] == -9999] = False
  good_dat[full_dat[n] < 0] = False
  good_dat[np.isnan(full_dat[n])] = False
  good_dat[np.isinf(full_dat[n])] = False

for n in range(0,len(full_dat)):
  print n
  full_dat[n] = full_dat[n][good_dat]
y_dat = y_dat[good_dat]


print('stacking....')
full_dat = np.transpose(np.vstack(full_dat))
print full_dat.shape
print('stacked')


print('permuting...')
np.random.seed(13)
perm = np.random.permutation(len(full_dat))
full_dat = full_dat[perm,:]
y_dat = y_dat[perm]
print('permuted')

print('saving....')
np.save('munged_dat/stack_dat_'+year+'_'+version,full_dat)
np.save('munged_dat/y_dat_'+year+'_'+version,y_dat)







