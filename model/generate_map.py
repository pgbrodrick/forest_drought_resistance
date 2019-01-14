#### Created by: Philip Brodrick
#### Purpose: Generate a CWC map for a specified year from a pre-trained model
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
#from pandas.io.pytables import HDFStore
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import cross_validation, metrics
import numpy.matlib
import time
import timeit

from sklearn.externals import joblib

import h5py
import argparse

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



def network(ws):
    model = Sequential()
    act = 'softplus'
    model.add(Dense(200, input_dim=full_dat.shape[1], kernel_initializer='normal'))
    model.add(Activation(act))
    model.add(Dense(200, kernel_initializer='normal'))
    model.add(Activation(act))
    model.add(Dense(200, kernel_initializer='normal'))
    model.add(Activation(act))
    model.add(Dense(200, kernel_initializer='normal'))
    model.add(Activation(act))
    model.add(Dense(1, kernel_initializer='normal'))
 
    model.load_weights('trained_models/' + version + '_epoch_35')
    #print 'compiling'
    model.compile(loss='mse',optimizer='adam')
    return model

def get_conv_dat_shape(read_dat_shape,ws):
  mmm = int(round(ws/2.)-1)  
  nnn = int(round(ws/2.)-1)  
  return (read_dat_shape[0]-ws,read_dat_shape[1]-ws)




parser = argparse.ArgumentParser(description='Train CWC model on GPUs with spatial holdout sets')
parser.add_argument('year')
parser.add_argument('-month',default='7')
parser.add_argument('-fold',type=int,default=-1)
parser.add_argument('-gpu',type=int,default=0)
args = parser.parse_args()

year = args.year
month = args.month
fold = args.fold
gpu = args.gpu


os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


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

filenames.append('dat/2017_refl/proc/california_sr_' + year + '_' + month + '.dat')


version = '2017cwc'
ws = 3
data_version = 'thresh10_ws' + str(ws)
use_xy = True
if(use_xy):
  version = version + '_xy'
else:
  filenames.pop(0)
  filenames.pop(0)
version = version + '_ws_' + str(ws) + '_' + str(fold)
xscaler = joblib.load('trained_models/' + version + '_xscaler')
yscaler = joblib.load('trained_models/' + version + '_yscaler')

dataset = gdal.Open(filenames[0],gdal.GA_ReadOnly)

y_chunks = np.linspace((ws-1)/2,dataset.RasterYSize - (ws-1)/2,num=100).astype(int)
pred_y = np.ones((dataset.RasterYSize,dataset.RasterXSize))*-9999
good_dat = np.ones((dataset.RasterYSize,dataset.RasterXSize)).astype(bool)

model_loaded = False
print(int(year))
for y_ind in range(0,len(y_chunks)-2):
  i_ymin = y_chunks[y_ind]
  i_ymax = y_chunks[y_ind+1]
  
  print(round(y_ind / float(len(y_chunks)-2),3))
  full_dat = []
  n_col = 0
  for fi in filenames:
   dataset = gdal.Open(fi,gdal.GA_ReadOnly)
   if ('/x.tif' in fi or '/y.tif' in fi):
    band_dat = dataset.GetRasterBand(1).ReadAsArray(0,int(i_ymin),int(dataset.RasterXSize),int(i_ymax-i_ymin+ws))
    band_dat = get_conv_dat(band_dat,ws,True)
    full_dat.append(band_dat)
    n_col+=1
   else:
    if (int(year) < 2013 and 'california_sr' in fi):
      b_list = list(range(1,dataset.RasterCount+1))
      b_list.pop(5)
      for b in b_list:
       read = dataset.GetRasterBand(b).ReadAsArray(0,int(i_ymin),int(dataset.RasterXSize),int(i_ymax-i_ymin+ws))
       ld = get_conv_dat(read,ws)
       for iii in range(0,len(ld)):
        if ('aspect' in fi):
         full_dat.append(np.sin(ld[iii]*np.pi/180.))
        else:
         full_dat.append(ld[iii])
        n_col+= 1
    else:
     for b in range(1,dataset.RasterCount+1):
      read = dataset.GetRasterBand(b).ReadAsArray(0,int(i_ymin),int(dataset.RasterXSize),int(i_ymax-i_ymin+ws))
      ld = get_conv_dat(read,ws)
      for iii in range(0,len(ld)):
       if ('aspect' in fi):
        full_dat.append(np.sin(ld[iii]*np.pi/180.))
       else:
        full_dat.append(ld[iii])
       n_col+= 1

  #print 'stacking....',
  lgd = np.ones(full_dat[0].shape).astype(bool)
  for n in range(0,n_col):
    lgd[full_dat[n] == -9999] = False
    lgd[np.isnan(full_dat[n])] = False
    lgd[np.isinf(full_dat[n])] = False
  full_dat = np.transpose(np.vstack(full_dat))

  lgd = lgd.flatten()
  if (np.sum(lgd) > 0):
    full_dat[lgd,:] = xscaler.transform(full_dat[lgd,:])
    
    if (model_loaded == False):
      model = network(ws)
      model_loaded = True
    
    # Parameters
    chunksize = 5000
    
    lpy = model.predict(full_dat,batch_size=chunksize*full_dat.shape[1],verbose=0)
    lpy = yscaler.inverse_transform(lpy)

  else:
    lpy = np.zeros((len(full_dat),2))-9999

  lpy[np.logical_not(np.transpose(lgd))] = -9999
  lpy = np.reshape(lpy[:,0],get_conv_dat_shape(read.shape,ws))

  pred_y[i_ymin:i_ymax,0:dataset.RasterXSize-ws] = lpy 


 

##### print final map

#pred_y = np.reshape(pred_y,(dataset.RasterYSize, dataset.RasterXSize))
driver = gdal.GetDriverByName('ENVI') 
driver.Register()
outname = os.getcwd() + '/maps/' + version + '_' + year + '_' + month
outDataset = driver.Create(outname,pred_y.shape[1],pred_y.shape[0],1,GDT_Float32)
outDataset.SetProjection(dataset.GetProjection())


outDataset.SetGeoTransform(dataset.GetGeoTransform())
outDataset.GetRasterBand(1).WriteArray(pred_y,0,0)
outDataset.GetRasterBand(1).SetNoDataValue(-9999)
del outDataset
 
print('printed')



