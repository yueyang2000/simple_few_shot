import os 
import numpy as np
import torch
from tqdm import tqdm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from model import BackBone
from dataset import Caltech256 

net = BackBone()

def get_all_features(split='train', load=True):
    dset = Caltech256(split=split)
    path = './data/'+split+'.npy'
    if load and os.path.exists(path):
        return np.load(path), dset.get_labels()
    
    features = []
    for idx in tqdm(range(len(dset))):
        features.append(net(dset[idx][0][None, :]).squeeze().numpy())
    features = np.asarray(features)
    if load:
        np.save(path, features)
    return features, dset.get_labels()

if __name__ == '__main__':
    print('Machine Learning Baselines')
    X_train, y_train = get_all_features(split='train')
    X_test, y_test = get_all_features(split='test')
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, y_train)
    print('KNN:', clf.score(X_test, y_test)) # 0.212
    clf = SVC()
    clf.fit(X_train, y_train)
    print('SVC:', clf.score(X_test, y_test)) # 0.548
