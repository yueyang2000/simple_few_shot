import os 
from PIL import Image 
import numpy as np 
import torch
from torchvision import transforms
from torchvision.transforms.functional import hflip, rgb_to_grayscale
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import torch

class Caltech256(VisionDataset):
    def __init__(self, root_dir='./data/256_ObjectCategories', split='train', transform=None):
        self.num_classes = 50
        self.split = split
        if self.split == 'train':
            self.n_shot=10
        else:
            self.n_shot=30

        self._read_all(root_dir)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.seed = np.random.RandomState(0)
        for i in range(len(self.data)):
            self.data[i] = self.transform(self.data[i])
        self.data = torch.stack(self.data)

    def _read_all(self, root_dir):
        self.data = [None for _ in range(50)]
        self.labels = [None for _ in range(50)]
        self.label_names = []
        for dir in os.listdir(root_dir):
            #print(dir)
            label, name = dir.split('.')
            label = int(label) - 1
            if label >= 50:
                continue
            self.label_names.append(name)
            sub_dir = os.path.join(root_dir, dir)
            cls_data = []
            cls_labels = []
            for img_name in os.listdir(sub_dir):
                img = Image.open(os.path.join(sub_dir, img_name)).copy().convert('RGB')
                cls_data.append(img)
                cls_labels.append(label)
                if len(cls_data) > 40: break
            if self.split == 'train':
                self.data[label] = cls_data[:10]
                self.labels[label] = cls_labels[:10]
            else:
                self.data[label] = cls_data[10:40]
                self.labels[label] = cls_labels[10:40]
        # sort in order
        data = []
        labels = []
        for i in range(50):
            data += self.data[i]
            labels += self.labels[i]
        self.data = data
        self.labels = torch.tensor(np.asarray(labels))
        

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def get(self, class_id, sample_id):
        return self.__getitem__(class_id * self.n_shot + sample_id)
    
    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def full_proto_batch(self):
        return self.data 
    
class Caltech256Aug(VisionDataset):
    def __init__(self, root_dir='./data/256_ObjectCategories', split='train'):
        self.num_classes = 50
        self.split = split


        self._read_all(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data = []
        labels = []
        if split == 'train':    
            for i in range(len(self.data)):
                data.append(self.transform(self.data[i]))
                data.append(self.transform(hflip(self.data[i])))
                data.append(self.transform(rgb_to_grayscale(self.data[i], 3)))
                data.append(self.transform(rgb_to_grayscale(hflip(self.data[i]), 3)))
                labels += [self.labels[i],self.labels[i],self.labels[i],self.labels[i]]
        else:
            for i in range(len(self.data)):
                data.append(self.transform(self.data[i]))
                labels += [self.labels[i]]
        self.seed = np.random.RandomState(0)

        self.data = torch.stack(data)
        self.labels = torch.tensor(np.asarray(labels))

    def _read_all(self, root_dir):
        self.data = [None for _ in range(50)]
        self.labels = [None for _ in range(50)]
        self.label_names = []
        for dir in os.listdir(root_dir):
            #print(dir)
            label, name = dir.split('.')
            label = int(label) - 1
            if label >= 50:
                continue
            self.label_names.append(name)
            sub_dir = os.path.join(root_dir, dir)
            cls_data = []
            cls_labels = []
            for img_name in os.listdir(sub_dir):
                img = Image.open(os.path.join(sub_dir, img_name)).copy().convert('RGB')
                cls_data.append(img)
                cls_labels.append(label)
                if len(cls_data) > 40: break
            if self.split == 'train':
                self.data[label] = cls_data[:10]
                self.labels[label] = cls_labels[:10]
            else:
                self.data[label] = cls_data[10:40]
                self.labels[label] = cls_labels[10:40]
        # sort in order
        data = []
        labels = []
        for i in range(50):
            data += self.data[i]
            labels += self.labels[i]
        self.data = data
        self.labels = labels
        
        

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def full_proto_batch(self):
        return self.data 

class Proj2Test(VisionDataset):
    def __init__(self, root_dir='./data/proj2_test_data'):
        self.data = [None for _ in range(2500)]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._read_all(root_dir)

    def _read_all(self, root_dir):
        for img_name in os.listdir(root_dir):
            idx = int(img_name.split('_')[1].split('.')[0]) - 1
            path = os.path.join(root_dir, img_name)
            img = Image.open(path).copy().convert('RGB')
            self.data[idx] = img

    def __getitem__(self, index: int):
        return self.transform(self.data[index])
    
    def __len__(self):
        return 2500
        
class ProtoBatchSampler(object):
    def __init__(self, n_sample, n_support, n_class, n_episode):
        super().__init__()
        self.n_sample = n_sample
        self.n_class = n_class 
        self.n_support = n_support
        self.n_episode = n_episode
        self.seed = np.random.RandomState(0)
    
    def __iter__(self):
        for _ in range(self.n_episode):
            choices = self.seed.permutation(50)[:self.n_class]
            support = []
            query = []
            for k in choices:
                perm = self.seed.permutation(np.arange(self.n_sample))
                support_choices = perm[:self.n_support]
                query_choices = perm[self.n_support:]
                support.append(k*self.n_sample + support_choices)
                query.append(k*self.n_sample + query_choices)
            yield torch.tensor(np.concatenate(support + query))
    
    def __len__(self):
        return self.n_episode



class BaseFeature:
    def __init__(self, root_dir='./data/', num_classes=50, n_sample=10):
        self.num_classes = num_classes
        self.n_sample = n_sample
        self.data = torch.tensor(np.load(os.path.join(root_dir, 'base_feature.npy')))
        self.labels = []
        if os.path.exists(os.path.join(root_dir, 'base_label.npy')):
            self.labels = np.load(os.path.join(root_dir, 'base_label.npy'))
        else:
            with open(os.path.join(root_dir, 'base_label.txt'), 'r') as f:
                for line in f.readlines():
                    self.labels.append(int(line.strip()) - 1)
            self.labels = np.asarray(self.labels)
            np.save(os.path.join(root_dir, 'base_label.npy'), self.labels)
        self.seed = np.random.RandomState(0)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]
    
    def sample_proto_batch(self, n_support):
        support = []
        query = []
        for k in range(self.num_classes):
            indices = np.where(self.labels == k)[0][:self.n_sample]
            perm = self.seed.permutation(indices)
            support.append(perm[:n_support])
            query.append(perm[n_support:])
        support = np.concatenate(support)
        query = np.concatenate(query)
        return torch.tensor(self.data[support]), torch.tensor(self.data[query])

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

class BaseFeatureDataset(Dataset):
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
        return torch.from_numpy(self.feature[index]), torch.from_numpy(self.label)

    def __len__(self):
        return 0

if __name__ == '__main__':
    #dataset = Caltech256()
    dataset = BaseFeature()
    support, query = dataset.sample_proto_batch(10,3,2)
