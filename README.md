# simple_few_shot

Few shot learning on Caltech256. Project of *Digital Image Processing (THU 2021)*

## Method

-  Loss for training: [Prototypical Loss](https://arxiv.org/abs/1703.05175) and [NCA Loss](https://arxiv.org/abs/1808.04699)
- Evaluation: 1-NN with centroid(Prototypical), k-NN, soft assignment (see [this paper](https://arxiv.org/abs/2012.09831))
- Distance metric: euclidean, cosine

## Results

We start with different hidden layers in the backbone (denote with layer1, layer2, layer3)

```python
# classifier of AlexNet
self.classifier = nn.Sequential(
    # layer 3
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    # layer 2
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    # layer 1
    nn.Linear(4096, num_classes),
)
```

 Evaluate directly **without training**

|         | SVM  | Proto(cos) | KNN(cos) | Soft(cos) | Proto(L2) | KNN(L2) | Soft(L2) |
| ------- | ---- | ---------- | -------- | --------- | --------- | ------- | -------- |
| Layer 1 | 58.0 | 67.27      | 57.73    | 64.2      | 64.87     | 43.53   | 2.07     |
| Layer 2 | 58.9 | **68.47**  | 59.47    | 62.47     | 64.93     | 37.67   | 2.07     |
| Layer 3 | 52.6 | 65.07      | 55.4     | 58.73     | 58.8      | 15.33   | 2.07     |

**Train a liear projection** on top of backbone, eval with prototypical(cosine)

| Eval Directly | Eval with Aug | Fine Tune | Proto Loss | NCA Loss |
| ------------- | ------------- | --------- | ---------- | -------- |
| 68.47         | 69.0          | 68.8      | **69.87**  | 69.07    |

## Data Preparation

Download `256_ObjectCategories.tar` and extract to `./data/256_ObjectCategories/`

Download pretrained AlexNet [checkpoint](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth) and save it to `./pretrained/alexnet-owt-4df8aa71.pth`

## Usage

- Baseline KNN and SVM classifier: `python3 baseline.py`
- Finetune Alexnet: `python3 finetune.py`
- Main script: `python3 train.py` (No augmentation), `python3 train_aug.py` (With augmentation)
  - key argument `--loss_type`：`proto` OR `nca`
  - key argument `--dist_type`: `euclidean` OR `cosine`
  - key argument `--test_bb`: whether to evaluate directly with
  - key argument `--layer_bb`: location of the hidden feature

- `model_pairs.py, MRN.py` are some failed trials to reproduce the [Model Regression](https://www.ri.cmu.edu/pub_files/2016/10/yuxiongw_eccv16_learntolearn.pdf) method

