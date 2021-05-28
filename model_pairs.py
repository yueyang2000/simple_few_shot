import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torch.optim as optim
from sklearn import svm
from sklearn.metrics import mean_squared_error

import argparse

from model import ModelRegression, BackBone
from dataset import Caltech256
from baseline import get_all_features

import warnings
warnings.filterwarnings("ignore")

N = 10
S = 5
M = 100
rps = [-2,-1,0,1,2]

class BaseFeature:
    def __init__(self, split='base'):
        self.categorys2feature = {}
        feature_path = './data/'+split+'_feature.npy'
        label_path = './data/'+split+'_label.txt'
        if os.path.exists(feature_path) and os.path.exists(feature_path):
            self.features = np.load(feature_path)  # (100000, 4096)
            self.labels = []
            with open(label_path) as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    label = int(line)
                    self.labels.append(label)
                    if label in self.categorys2feature:
                        self.categorys2feature[label].append(self.features[idx])
                    else:
                        self.categorys2feature[label] = [self.features[idx]]
            self.labels = np.asarray(self.labels)  # (100000,)
            self.len = len(self.labels)
            print("{} categorys".format(len(self.categorys2feature.items() ) ))

    def get_base_features(self):
        return self.categorys2feature

    def get_neg_c_random_features(self, c, size):
        features = []
        for i in range(size):
            idx = random.randint(0,self.len-1)
            while self.labels[idx] == c:
                idx = random.randint(0,self.len-1)
            features.append(self.features[idx])
        return features

def generate_model_pairs(path):
    base_feature = BaseFeature()
    modle_pairs = []

    for c, c_feature in tqdm(base_feature.get_base_features().items()):
        # for each category c
        w_list = []  # w* , w0, w0, w0, ...
        Y = np.ones( (len(c_feature),), dtype=np.int16)
        Y = np.append(Y, -np.ones( (M,), dtype=np.int16))
        neg_c_features = base_feature.get_neg_c_random_features(c, M)
        c_feature += neg_c_features

        # large
        # print('===large===')
        clf = svm.LinearSVC(max_iter=5000)
        # print(np.asarray(c_feature).shape, Y.shape)
        clf.fit(np.asarray(c_feature), Y)
        # w_star1 = clf.coef_
        # w_star2 = clf.intercept_
        w_star = np.append(np.append(clf.coef_, clf.intercept_), c)
        # w_star3 = clf.classes_  # (2,) [-1  1]
        # print("w_star.shape: ", w_star.shape)
        w_list.append(w_star)

        # small (size = 10)
        # print('===small===')
        Y_small = np.ones( (N,), dtype=np.int16)
        Y_small = np.append(Y_small, -np.ones( (M,), dtype=np.int16))
        for i in range(S):
            c_features_small = []
            random_idx = []
            for j in range(N):
                idx = random.randint(0,len(c_feature)-1)
                while idx in random_idx:
                    idx = random.randint(0,len(c_feature)-1)
                random_idx.append(idx)
            for idx in random_idx:
                c_features_small.append(c_feature[idx])
            c_features_small += neg_c_features

            for rp in rps:  # regularization parameters
                clf_small = svm.LinearSVC(max_iter=5000, C=10**rp)
                # print(np.asarray(c_features_small), Y_small)
                # print(np.asarray(c_features_small).shape, Y_small.shape)
                clf_small.fit(np.asarray(c_features_small), Y_small)
                # w_0_1 = clf_small.coef_
                # w_0_2 = clf_small.intercept_
                w_0 = np.append(np.append(clf_small.coef_, clf_small.intercept_), c)
                # print('w_0.shape: ', w_0.shape)
                w_list.append(w_0)

        w_list = np.asarray(w_list)
        modle_pairs.append(w_list)

    modle_pairs = np.asarray(modle_pairs)
    print("modle_pairs.shape: ", modle_pairs.shape)
    np.save(path, modle_pairs)

def get_w0(path):
    features, labels = get_all_features(split='train')
    size = len(labels)
    idx = 0
    w_dict = {}  # label : w0
    while(idx < size):
        label = labels[idx:idx+10]
        feature = features[idx:idx+10]
        idx += 10


        Y = np.ones( (feature.shape[0]), dtype=np.int16)
        Y = np.append(Y, -np.ones( (M,), dtype=np.int16))
        neg_c_features = []
        for i in range(M):
            j = random.randint(0,len(features)-1)
            while labels[j] == label[0]:
                j = random.randint(0,len(features)-1)
            neg_c_features.append(features[j])
        neg_c_features = np.asarray(neg_c_features)

        feature = np.concatenate((feature,neg_c_features),axis=0)

        clf = svm.LinearSVC(max_iter=5000)
        # print(feature.shape, Y.shape)
        clf.fit(feature, Y)
        w0 = np.append(clf.coef_, clf.intercept_)
        # print("w0.shape: ", w0.shape)
        w_dict[label[0]] = w0

    w_list = [(w_dict[k]) for k in sorted(w_dict.keys())]

    w_list = np.asarray(w_list)
    print("w_list.shape: ", w_list.shape)
    np.save(path, w_list)


if __name__ == '__main__':
    # print('Generate modelpairs')
    # generate_model_pairs('./data/modelpairs.npy')
    print('Generate Caltech256_w0.npy')
    get_w0("./data/Caltech256_w0.npy")