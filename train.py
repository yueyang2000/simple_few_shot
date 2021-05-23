import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torch.optim as optim
from sklearn import svm

import argparse

from model import ModelRegression, BackBone

M = 100

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
                    self.labels.append(int(line))
                    if self.categorys2feature.get(idx) == None:
                        self.categorys2feature[int(line)] = [self.features[idx]]
                    else:
                        self.categorys2feature[int(line)].append(self.features[idx])
            self.labels = np.asarray(self.labels)  # (100000,)

    def get_base_features(self):
        return self.categorys2feature

    def get_neg_c_random_features(self, c, size):
        pass
        # TODO



def generate_model_pairs():
    base_feature = BaseFeature()

    # Initialize SVM classifier
    clf = svm.LinearSVC()

    for c, c_feature in base_feature.get_base_features.items():
        # for each category c
        features = c_feature
        Y = np.ones( (len(c_feature),), dtype=np.int16)
        features.append(base_feature.get_neg_c_random_features(c, M))
        Y = np.append(Y, -np.ones( (M,), dtype=np.int16))
        # large
        clf.fit(features, Y)
        # TODO: get w_star

        # small (size = 10)
        for i in range(S):

            clf.fit(X_train_small, y_train_small)
        # TODO: get w_0

    return



if __name__ == '__main__':
    print('Train regression function T')
    generate_model_pairs()