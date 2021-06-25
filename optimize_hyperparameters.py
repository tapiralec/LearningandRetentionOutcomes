import sys

#sys.path.append('..')
import numpy as np
import pickle
import os
import agm
import optuna
from itertools import product, chain
from lgp.classification.tune_hyper import sequence_accuracy, get_score
from sklearn.metrics import classification_report
from agm.userlevelml import GridUserLevelSearchCV, train_test_user_stratified_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tqdm import tqdm

RANDSTATE = 2021
VERBOSE = 0

# specify what columns we'll want out of the tracking data
trackers = ['Head Transform', 'Left hand Transform', 'Right hand Transform']
vtrackers = ['v Head', 'v LHand', 'v RHand']
pos_only = ['X', 'Y', 'Z']
pos_and_rot = pos_only + ['w', 'x', 'y', 'z']

attrib_options = [[" ".join(c) for c in product(trackers, pos_only)],
                  [" ".join(c) for c in chain(product([trackers[0]], pos_and_rot), product(trackers[1:], pos_only))],
                  [" ".join(c) for c in product(trackers, pos_and_rot)]]
vattrib_options = [[" ".join(c) for c in product(vtrackers, pos_only)],
                   [" ".join(c) for c in chain(product([vtrackers[0]], pos_and_rot), product(vtrackers[1:], pos_only))],
                   [" ".join(c) for c in product(vtrackers, pos_and_rot)]]

# specify what values we'll want to predict
searchable_y = ['Learning Success', 'Knowledge Test Score', 'Practice Success', 'Retention Test Score']
y_cutoffs = [.9321, 12, .5875, 12]


# def round(use_position,x_sel,y_sel,win_shift,win_size,use_pca,feature_para,svm_c,svm_gamma):
def objective(trial):
    # win_shift = [5,7,9,11,13]
    win_shift = trial.suggest_int("win_shift", 5, 13, 2)
    # win_size = [30,45,60,75,90,105,120]
    win_size = trial.suggest_int("win_size", 30, 120, 15)
    # feature_para = [10, 20, 30, 40, 50, 60, 70, 80]
    feature_para = trial.suggest_int("feature_para", 10, 80, 10)
    # svm_C = [1,10,100,1000,10000]
    svm_C = trial.suggest_categorical("svm_c", [1, 10, 100, 1000, 10000])
    # svm_gamma = [b*10**p for p in range(-3,-1) for b in [1,3,5,7,9]]
    svm_gamma = trial.suggest_categorical("svm_gamma", [b * 10 ** p for p in range(-3, -1) for b in [1, 3, 5, 7, 9]])

    motion_type_str = 'Position' if use_position else 'Velocity'
    precachedFile = os.path.join(f'cache_windows/Learning{motion_type_str}', f'sliding{str((win_shift, win_size))}.pkl')
    cachedX, cachedY, cachedusers = pickle.load(open(precachedFile, 'rb'))
    if use_position:
        inputX = np.vstack([_x[attrib_options[x_sel]].values.flatten() for _x in cachedX])
    else:
        inputX = np.vstack([_x[vattrib_options[x_sel]].values.flatten() for _x in cachedX])
    inputY = np.greater_equal(np.array([_y[searchable_y[y_sel]] for _y in cachedY]), y_cutoffs[y_sel]).astype(np.int64)
    inputusers = np.unique(cachedusers, return_inverse=True)[1]
    trainX, testX, trainy, testy, trainusers, testusers = train_test_user_stratified_split(inputX, inputY, inputusers,
                                                                                           train_size=0.8,
                                                                                           random_state=RANDSTATE)
    if VERBOSE>2:
        for uid in np.unique(trainusers):
            if trainy[trainusers == uid][0] == 1:
                print(f'train high: {uid}')
        for uid in np.unique(testusers):
            if testy[testusers == uid][0] == 1:
                print(f'test high: {uid}')

    # now we have the train/test splits, we need to create the k-folds and evaluate on those

    inner4fold = agm.userlevelml.StratifiedUserKFold(4)
    splits = inner4fold.split(trainX, trainy, trainusers)
    result = []
    for split in tqdm(splits, total=4, leave=False):
        if VERBOSE>1:
            print(f'training on {len(np.unique(trainusers[split[0]]))}, testing on {len(np.unique(trainusers[split[1]]))}')
        ktrainX = trainX[split[0]]
        ktestX = trainX[split[1]]
        ktrainy = trainy[split[0]]
        ktesty = trainy[split[1]]
        ktrainusers = trainusers[split[0]]
        ktestusers = trainusers[split[1]]

        fea_func = PCA(n_components=feature_para)

        # run component analysis on train subset only, apply to all:
        fea_func.fit(ktrainX)
        featrainX = fea_func.transform(ktrainX)
        featestX = fea_func.transform(ktestX)

        svc = SVC(class_weight='balanced', C=svm_C, gamma=svm_gamma)
        svc.fit(featrainX, ktrainy)
        predy = svc.predict(featestX)
        # svc.decision_function(testX)

        report = classification_report(ktesty, predy, output_dict=True)
        report['seq_perf'] = sequence_accuracy(svc, (featestX, ktesty, ktestusers, None))
        # print(report)
        result.append(report)
        for user in ktestusers:
            sel = user == ktestusers
            userpred = predy[sel]
            useractual = ktesty[sel]

    if VERBOSE>3:
        for rprt in result:
            # print(result[0]['seq_perf'])
            info = rprt['seq_perf']
            print(info)
            novice = info[0]
            exp = info[1]
    score = get_score(result)
    return score

#%%
def canonicalstudyname(ysel,xsel,pcacvx,posvel):
    yname = ['LS','KT','PS','PT'][ysel]
    xname = ['norot','headrot','allrot'][xsel]
    pcaname = 'pca' if pcacvx else 'cvx'
    posname = 'pos' if posvel else 'vel'
    return f'{yname}_{xname}_{pcaname}_{posname}'

# %%
# Ideally this would be a callable function, I guess...
search_space = {
    "win_shift": [5, 7, 9, 11, 13],
    "win_size": [30, 45, 60, 75, 90, 105, 120],
    "feature_para": [10, 20, 30, 40, 50, 60, 70, 80],
    "svm_c": [1, 10, 100, 1000, 10000],
    "svm_gamma": [b * 10 ** p for p in range(-3, -1) for b in [1, 3, 5, 7, 9]],
}
# y_sel = 0-3 -- which value to try to predict
# ['Learning Success', 'Knowledge Test Score', 'Practice Success', 'Retention Test Score']
y_sel = 0
# x_sel = 0-2 -- which features to use
# 0 = no rot, 1 = head rot, 2 = head+hands rot (allrot)
x_sel = 0  # trial.suggest_categorical("x_sel",[0,1,2])
# use_pca = True if using PCA, False if using CVX (not implemented)
use_pca = True
# use_position = True if using position, False if using velocity
use_position = True



for ysel in [2,3,1]:
    for xsel in [0,1,2]:
        for posvel in [True, False]:
            # using pca only.
            y_sel = ysel
            x_sel = xsel
            use_position = posvel

            study_name = canonicalstudyname(y_sel,x_sel,use_pca,use_position)
            print(study_name)
            #study_name = "LS_norot_pca_vel"
            storage_name = "sqlite:///{}.db".format(study_name)

            study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize',
                                        load_if_exists=True, sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective)


