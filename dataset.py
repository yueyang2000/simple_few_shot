import os 
from PIL import Image 
import numpy as np 
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class Caltech256(VisionDataset):
    def __init__(self, root_dir='./data/training', split='train',n_shot=5, transform=None):
        self.n_shot = n_shot
        self.split = split
        self._read_all(root_dir)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def _read_all(self, root_dir):
        self.data = []
        self.labels = []
        self.label_names = []
        for dir in os.listdir(root_dir):
            label, name = dir.split('.')
            label = int(label) - 1
            self.label_names.append(name)
            sub_dir = os.path.join(root_dir, dir)
            cls_data = []
            cls_labels = []
            for img_name in os.listdir(sub_dir):
                img = Image.open(os.path.join(sub_dir, img_name)).copy().convert('RGB')
                cls_data.append(img)
                cls_labels.append(label)
            if self.split == 'train':
                self.data += cls_data[:self.n_shot]
                self.labels += cls_labels[:self.n_shot]
            else:
                self.data += cls_data[self.n_shot:]
                self.labels += cls_labels[self.n_shot:]

        self.labels = np.asarray(self.labels)

    def __getitem__(self, index: int):
        return self.transform(self.data[index]), self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels


class ModelPair(Dataset):
    def __init__(self):
        super().__init__()
        modelpair_path = './data/modelpairs.npy'
        self.load_data(modelpair_path)

    def load_data(self, pairs_path):
        self.modelpairs = np.load(pairs_path)    # (1000, 26, 4098)
        self.classes = self.modelpairs.shape[0]
        self.pairs = self.modelpairs.shape[1] - 1
        self.len = self.classes * self.pairs  # 25k
    
    # return w0, w*, class_index
    def __getitem__(self, index):
        n = index // self.pairs
        p = index % self.pairs + 1
        return self.modelpairs[n][p][0:-1], self.modelpairs[n][0][0:-1], self.modelpairs[n][0][-1]
    
    def __len__(self):
        return self.len


class BaseFeature(Dataset):
    def __init__(self):
        super().__init__()
        self.N = 10
        self.M = 50
        svm_feature = './data/svm_feature.npy'
        if os.path.exists(svm_feature):
            self.feature = np.load(svm_feature)
            self.label = np.append(np.ones(self.N), -np.ones(self.M))
        else:
            feature_path = './data/base_feature.npy'
            label_path = './data/base_label.txt'
            self.load_data(feature_path, label_path)
            np.save('./data/svm_feature.npy', self.feature)
    
    def load_data(self, feature_path, label_path):
        with open(label_path) as f:
            labels = f.readlines()
        features = np.load(feature_path)
        N = self.N
        M = self.M
        self.feature = np.zeros((1000, N+M, 4097))
        self.label = np.append(np.ones(N), -np.ones(M))
        print('generate features for svm')
        for i in tqdm(range(1000)):   # 1000 class
            label = int(labels[i * 100]) - 1 # 100 items per class
            pos = features[i * 100: i * 100 + N]
            pos = np.append(pos, np.ones((N, 1)), axis=1) # append 1 to the end (for intercept)
            neg = []
            for j in range(M):
                idx = random.randint(0, 100000)
                while idx // 100 != label:
                    idx = random.randint(0, 100000)
                neg.append(features[idx])
            neg = np.array(neg)
            neg = np.append(neg, np.ones((M, 1)), axis=1)
            self.feature[label] = np.append(pos, neg, axis=0) 

    def __getitem__(self, index):
        return self.feature[index], self.label

    def __len__(self):
        return 0


if __name__ == '__main__':
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(256),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()
    # ])
    # dataset = Caltech256()
    # dataset = ModelPair()
    dataset = BaseFeature()
    