import os 
from PIL import Image 
import numpy as np 
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

class Caltech256(VisionDataset):
    def __init__(self, root_dir='./data/training', split='train',n_shot=5, transform=None):
        self.num_classes = 50
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
        self.seed = np.random.RandomState(0)

    def __getitem__(self, index: int):
        return self.transform(self.data[index]), self.labels[index]

    def get(self, class_id, sample_id):
        return self.__getitem__(class_id * self.n_shot + sample_id)
    
    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels
    
    def sample_proto_batch(self, n_class, n_support):
        choices = self.seed.permutation(50)[:n_class]
        support = []
        query = []
        for k in choices:
            perm = self.seed.permutation(np.arange(self.n_shot))
            support_choices = perm[:n_support]
            query_choices = perm[n_support:]
            for s in support_choices:
                support.append(self.get(k, s)[0])
            for q in query_choices:
                query.append(self.get(k, q)[0])
        return torch.stack(support), torch.stack(query)




if __name__ == '__main__':
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(256),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()
    # ])
    dataset = Caltech256()
    