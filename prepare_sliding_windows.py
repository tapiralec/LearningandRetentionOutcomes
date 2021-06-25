#%%

import pandas as pd
import os
from itertools import product, chain
import pickle
from tqdm.notebook import tqdm
from multiprocessing import Pool


import numpy as np
#from sklearn.preprocessing import

#%%

# specify what columns we'll want out of the tracking data
trackers = ['Head Transform','Left hand Transform','Right hand Transform']
vtrackers = ['v Head','v LHand','v RHand']
pos_only = ['X','Y','Z']
pos_and_rot = pos_only+['w','x','y','z']

attrib_options = [[" ".join(c) for c in product(trackers,pos_only)],
                  [" ".join(c) for c in chain(product([trackers[0]],pos_and_rot),product(trackers[1:],pos_only))],
                  [" ".join(c) for c in product(trackers,pos_and_rot)]]
vattrib_options = [[" ".join(c) for c in product(vtrackers,pos_only)],
                   [" ".join(c) for c in chain(product([vtrackers[0]],pos_and_rot),product(trackers[1:],pos_only))],
                   [" ".join(c) for c in product(vtrackers,pos_and_rot)]]

# specify what values we'll want to predict
searchable_y = ['Learning Success','Knowledge Test Score','Practice Success','Retention Test Score']
y_cutoffs = [.9321,12,.7375,12]

#%%

vattrib_options[2]

#%%

# precompute sliding windows for Learning Position (and subsample every 15 frames)
shift_size = [5,7,9,11,13]
segment_length = [30,45,60,75,90,105,120]

data_dir = 'data/Annotated_withcat/LearningPosition'
outcache = 'cache_windows/LearningPosition/'
STEPSIZE = 15
Ycols = ['Learning Success','Knowledge Test Score','Practice Success','Retention Test Score']
for f in tqdm(os.listdir(data_dir)):
    uid = f[:3]
    df = pd.read_csv(os.path.join(data_dir,f))
    df.insert(loc=1,column='PID',value=uid)
    rows = df.shape[0]
    for ssize,slength in tqdm(product(shift_size,segment_length),total=len(shift_size)*len(segment_length),leave=False):

        X,Y,users = [],[],[]
        pklcache = os.path.join(outcache,'sliding'+str((ssize,slength))+'.pkl')
        if os.path.exists(pklcache):
            X,Y,users = pickle.load(open(pklcache,'rb'))
        sl = int(slength*90)
        sh = int(ssize*90)
        for start in range(0,rows-sl+1,sh):
            #print(start,start+sl)
            #print(df.iloc[start:start+sl+1:STEPSIZE,:])
            thisseg = df.iloc[start:start+sl+1:STEPSIZE,:]
            _X = thisseg[[c for c in thisseg.columns if 'Transform' in c]]
            _Y = thisseg[Ycols].iloc[0]
            _users = thisseg['PID'].iloc[0]
            X.append(_X)
            Y.append(_Y)
            users.append(_users)
        pickle.dump((X,Y,users),open(pklcache,'wb'))

#%%

# do the same for learning velocity (note the data has already been subsampled, so use STEPSIZE=1
shift_size = [5,7,9,11,13]
segment_length = [30,45,60,75,90,105,120]

data_dir = 'data/Annotated_withcat/LearningVelocity'
outcache = 'cache_windows/LearningVelocity/'
STEPSIZE = 1
Ycols = ['Learning Success','Knowledge Test Score','Practice Success','Retention Test Score']
for f in tqdm(os.listdir(data_dir)):
    uid = f[:3]
    df = pd.read_csv(os.path.join(data_dir,f))
    df.insert(loc=1,column='PID',value=uid)
    rows = df.shape[0]
    for ssize,slength in tqdm(product(shift_size,segment_length),total=len(shift_size)*len(segment_length),leave=False):

        X,Y,users = [],[],[]
        pklcache = os.path.join(outcache,'sliding'+str((ssize,slength))+'.pkl')
        if os.path.exists(pklcache):
            X,Y,users = pickle.load(open(pklcache,'rb'))
        sl = int(slength*90/15)
        sh = int(ssize*90/15)
        for start in range(0,rows-sl,sh):
            #print(start,start+sl)
            #print(df.iloc[start:start+sl+1:STEPSIZE,:])
            thisseg = df.iloc[start:start+sl+1:STEPSIZE,:]
            _X = thisseg[[c for c in thisseg.columns if c in vattrib_options[2]]]
            if _X.shape==(180,21):
                print(thisseg['PID'].iloc[0],start,start+sl+1)
            _Y = thisseg[Ycols].iloc[0]
            _users = thisseg['PID'].iloc[0]
            X.append(_X)
            Y.append(_Y)
            users.append(_users)
        pickle.dump((X,Y,users),open(pklcache,'wb'))
