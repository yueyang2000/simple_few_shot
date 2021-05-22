import os 
import numpy as np
import torch
from tqdm import tqdm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from model import BackBone
from model import FineTuner
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
    # X_train, y_train = get_all_features(split='train')
    # X_test, y_test = get_all_features(split='test')
    # clf = KNeighborsClassifier(n_neighbors=10)
    # clf.fit(X_train, y_train)
    # print('KNN:', clf.score(X_test, y_test)) # 0.212
    # clf = SVC()
    # clf.fit(X_train, y_train)
    # print('SVC:', clf.score(X_test, y_test)) # 0.548

    model = FineTuner()

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_data = Caltech256(split='train')
    dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    epochs = 10
    model.train()
    for epoch in tqdm(range(epochs)):
        for index, (img, label) in enumerate(dataloader):
            pred = model(img)
            loss = criterion(pred, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # test
    test_data = Caltech256(split='test')
    dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=len(test_data))
    model.eval()
    with torch.no_grad():
        for index, (img, label) in enumerate(dataloader):
            pred = model(img)
            _, pred = torch.max(pred.data, 1)
            total = len(test_data)
            correct = (pred == label).sum().item()
            acc = correct / total
            print(f'Accuracy: {acc}')
