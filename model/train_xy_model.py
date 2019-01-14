#### Created by: Philip Brodrick
#### Purpose: Train a CWC model
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

import tensorflow as tf
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
import h5py
import argparse



def gen_weights(yd):
    hist = np.histogram(yd,10)
    hist[1][-1] += 0.1
    weights = np.zeros(len(yd))
    for n in range(0,len(hist[1])-1):
        weights[np.logical_and(yd >= hist[1][n],yd < hist[1][n+1])] = n
    weights = np.ones(len(yd))
    weights[yd > 15] = 2
    weights[yd > 20] = 5
    weights[yd > 30] = 10 
    return weights

def rint(num):
  return int(round(num))


parser = argparse.ArgumentParser(description='Train CWC model on GPUs with spatial holdout sets')
parser.add_argument('ws',type=int,default=1)
parser.add_argument('-fold',type=int,default=0)
parser.add_argument('-gpu',type=int,default=0)
parser.add_argument('-warm_start_epoch',type=int,default=0)

args = parser.parse_args()
ws = args.ws
fold = args.fold
gpu = args.gpu
warm_start_epoch = args.warm_start_epoch

np.random.seed(13)


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







version = '2017cwc'
data_version = 'thresh10_ws' + str(ws)
years = ['2015','2016','2017']
use_xy = True
if(use_xy):
  version = version + '_xy'
version = version + '_ws_' + str(ws) + '_' + str(fold)


train_X = []
train_Y = []
test_X = []
test_Y = []

x_min = 1e12
x_max = -1e12
y_min = 1e12
y_max = -1e12
npzf = 'munged_dat/yp_for_upscale_' + data_version + '_' + str(fold) + '.npz'
if (os.path.isfile(npzf) == False):
  for year in years:
    full_dat = np.load('munged_dat/stack_dat_'+year+'_'+data_version + '.npy')
    print(full_dat.shape)
    x_min = min(np.min(full_dat[:,0]) , x_min)
    x_max = max(np.max(full_dat[:,0]) , x_max)
    y_min = min(np.min(full_dat[:,1]) , y_min)
    y_max = max(np.max(full_dat[:,1]) , y_max)
      
  if (fold == -1):
   for year in years:
    if (year in years):
     print('loading stack')
     full_dat = np.load('munged_dat/stack_dat_'+year+'_'+data_version + '.npy')
     print('loading y')
     y_dat = np.load('munged_dat/y_dat_'+year+'_'+data_version + '.npy')
     train_X.append(full_dat)
     train_Y.append(y_dat)
  else:
    n_xstep = 100
    n_ystep = 100
    x_space = np.linspace(x_min-1,x_max,n_xstep)
    y_space = np.linspace(y_min-1,y_max,n_ystep)
    print((x_min,x_max,y_min,y_max,x_space[1]-x_space[0],y_space[1]-y_space[0]))
    grids = np.zeros((len(y_space),len(x_space))).flatten()
    grids[np.random.permutation(len(grids))[ rint(0.1*(fold)*len(grids))  :rint(0.1*(fold+1)*len(grids))]] = 1
    grids = grids.reshape((len(y_space),len(x_space)))

    for year in years:
     if (year in years):
      print('loading stack')
      full_dat = np.load('munged_dat/stack_dat_'+year+'_'+data_version + '.npy')
      print('loading y')
      y_dat = np.load('munged_dat/y_dat_'+year+'_'+data_version + '.npy')
    
      print(full_dat.shape)
      test_set = np.zeros(len(full_dat)).astype(bool)
      individual_sums = []
      for n in range(0,n_xstep-1):
       print(n)
       for m in range(0,n_ystep-1):
        #if (np.random.uniform(0,1) < 0.1):
        if(grids[m,n] == 1):
          valid = np.logical_and(full_dat[:,0] > x_space[n],full_dat[:,0] <= x_space[n+1])
          valid[full_dat[:,1] <= y_space[m]] = False
          valid[full_dat[:,1] > y_space[m+1]] = False
          test_set[valid] = True
          sum = np.sum(valid)
          if (sum > 0):
            individual_sums.append(sum)
 
      test_X.append(full_dat[test_set,:])
      test_Y.append(y_dat[test_set])

      train_X.append(full_dat[np.logical_not(test_set),:])
      train_Y.append(y_dat[np.logical_not(test_set)])
      print(('year ',year,np.sum(np.logical_not(test_set)),np.sum(test_set)))
      print(individual_sums)

    del full_dat
    np.savez(npzf,train_X=train_X,test_X=test_X,train_Y=train_Y,test_Y=test_Y) 
else:
 npzf = np.load(npzf) 
 train_X = npzf['train_X']
 train_Y = npzf['train_Y']
 test_X = npzf['test_X']
 test_Y = npzf['test_Y']


train_X = np.vstack(train_X)
train_Y = np.hstack(train_Y)
if (use_xy == False):
  train_X = train_X[:,2:]
print((train_X.shape))
print((train_Y.shape))

print(('train response mean: ',np.mean(train_Y)))
print(('train response std: ',np.std(train_Y)))


print(('permuting...'))
perm = np.random.permutation(len(train_X))
train_X = train_X[perm,:]
train_Y = train_Y[perm]
print(('permuted'))


xscaler = preprocessing.StandardScaler()
train_X = xscaler.fit_transform(train_X)
yscaler = preprocessing.StandardScaler()
train_Y = yscaler.fit_transform(train_Y)
#yscaler = yscaler.fit(train_Y)

if (fold != -1):
  for i in range(0,len(test_X)):
    #test_Y[i] = test_Y[i].reshape(-1,1)
    print((test_Y[i].shape))
    print((test_X[i].shape,len(test_X)))
    if (use_xy == False):
      test_X[i] = test_X[i][:,2:]
    if (len(test_X[i]) > 0):
      test_X[i] = xscaler.transform(test_X[i])
      test_Y[i] = yscaler.transform(test_Y[i].reshape(-1,1))
    else:
      test_Y[i] = test_Y[i].reshape(-1,1)
  
  comb_test_X = np.vstack(test_X)
  comb_test_Y = np.vstack(test_Y)

joblib.dump(xscaler,'trained_models/' + version + '_xscaler')
joblib.dump(yscaler,'trained_models/' + version + '_yscaler')

chunksize = 5000


model = Sequential()

act = 'softplus'
#act = 'relu'
#act = 'tanh'
model.add(Dense(200, input_dim=train_X.shape[1], kernel_initializer='normal'))
model.add(Activation(act))
model.add(Dense(200, kernel_initializer='normal'))
model.add(Activation(act))
model.add(Dense(200, kernel_initializer='normal'))
model.add(Activation(act))
model.add(Dense(200, kernel_initializer='normal'))
model.add(Activation(act))
model.add(Dense(1, kernel_initializer='normal'))

if (warm_start_epoch > 0):
  model.load_weights('trained_models/' + version + '_epoch_' + str(warm_start_epoch))
print('compiling')
model.compile(loss='mae',optimizer='adam')
print('compiled...starting fit')
for n in range(0,7):
  if (fold != -1):
    model.fit(train_X,train_Y,validation_data=(comb_test_X,comb_test_Y),epochs=5,batch_size=chunksize,verbose=1)
  else:
    model.fit(train_X,train_Y,epochs=5,batch_size=chunksize,verbose=1)
  model.save_weights('trained_models/'+ version + '_epoch_' + str((n+1)*5 + warm_start_epoch),overwrite=True)





