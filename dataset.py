import os 
from PIL import Image 
import numpy as np 
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

class Caltech256(VisionDataset):
    def __init__(self, root_dir='./data/training', split='train',n_shot=5):
        self.n_shot = n_shot
        self.split = split
        self._read_all(root_dir)
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

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


if __name__ == '__main__':
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(256),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()
    # ])
    dataset = Caltech256()
    