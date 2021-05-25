import os 
from PIL import Image 
import numpy as np 
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

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

    def _read_all(self, root_dir):
        self.data = []
        self.labels = []
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
                self.data += cls_data[:10]
                self.labels += cls_labels[:10]
            else:
                self.data += cls_data[10:40]
                self.labels += cls_labels[10:40]

        self.labels = np.asarray(self.labels)
        

    def __getitem__(self, index: int):
        return self.transform(self.data[index]), self.labels[index]

    def get(self, class_id, sample_id):
        return self.__getitem__(class_id * self.n_shot + sample_id)
    
    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels
    
    
    def sample_proto_batch(self, n_class, n_support):
        choices = self.seed.permutation(self.num_classes)[:n_class]
        support = []
        query = []
        for k in choices:
            perm = self.seed.permutation(np.arange(self.n_shot))
            support_choices = torch.tensor(perm[:n_support])
            query_choices = torch.tensor(perm[n_support:])
            for s in support_choices:
                support.append(self.data[k*self.n_shot + support_choices])
            for q in query_choices:
                query.append(self.get(k, q)[0])
        return torch.stack(support), torch.stack(query)

    def full_proto_batch(self):
        n_sample = len(self.data) // self.num_classes
        support = []
        for k in range(self.num_classes):
            for s in range(n_sample):
                support.append(self.get(k,s)[0])
        return torch.stack(support)


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
    def __init__(self, root_dir='./data/'):
        self.num_classes = 1000
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
    
    def sample_proto_batch(self, n_class, n_support, n_query):
        choices = self.seed.permutation(self.num_classes)[:n_class]
        support = []
        query = []
        for k in choices:
            indices = np.where(self.labels == k)[0]
            perm = self.seed.permutation(indices)
            for s in perm[:n_support]:
                support.append(self.data[s])
            for q in perm[n_support:n_support+n_query]:
                query.append(self.data[q])
        return torch.stack(support), torch.stack(query)

if __name__ == '__main__':
    #dataset = Caltech256()
    dataset = BaseFeature()
    support, query = dataset.sample_proto_batch(10,3,2)