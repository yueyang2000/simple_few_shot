# simple_few_shot

Few shot learning on Caltech256. Project of *Digital Image Processing (THU 2021)*

## User Guide

Download experiment materials and rename the folder to `data`

Download pretrained AlexNet [checkpoint](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth) and save it to `pretrained/alexnet-owt-4df8aa71.pth`

## Contents

- `baseline.py`: Baseline KNN and SVM classifier

- `model.py`: network definitions

- `dataset.py`: a simple loader of Caltech256