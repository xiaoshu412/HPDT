import collections
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import random
from torchtoolbox.transform import Cutout
# %%
# 参数设置
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ISIC2018Dataset():
    """ ISIC 2018 Dataset
    Args:
        csv_path(str): csv file path of ISIC 2018.
        img_path(str): image folder of ISIC 2018.
        transform: image transform option.
    """

    def __init__(self, csv_file_path: str, img_dir: str,
                 transform=None, target_transform=None, **kwargs):
        super(ISIC2018Dataset, self).__init__(**kwargs)
        self.img_dir = img_dir
        self.trans = transform
        self.target_trans = target_transform

        df = pd.read_csv(csv_file_path)
        self.target_to_label = list(df.columns.values)[1:]
        self.label_to_target = {
            label: target for target, label in enumerate(self.target_to_label)
        }
        arr = np.array(df[self.target_to_label])

        self.img_names = list(df['image'])
        self.targets = list(arr.argmax(axis=1))
        self.categories = [*self.target_to_label]
        self.num_classes = len(self.target_to_label)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index] + '.jpg')
        image = Image.open(img_path)
        target = self.targets[index]

        if self.trans is not None:
             image = self.trans(image)

        if self.target_trans is not None:
            target = self.target_trans(target)

        return image, target
    def count_samples(self) -> list:
        """ count sample_nums """
        counter = collections.Counter(self.targets)
        class_nums = [(label, counter[target]) for target, label in enumerate(self.target_to_label)]
        return class_nums

    def to_targets(self, label_list: list) -> list:
        targets = [self.label_to_target[i] for i in label_list]
        return targets

    def to_labels(self, target_list: list) -> list:
        labels = [self.target_to_label[int(i)] for i in target_list]
        return labels
from torchvision import transforms
class MixUpTransform:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def __call__(self, image):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = image.size()[0]
        index = torch.randperm(batch_size)
        mixed_image = lam * image + (1 - lam) * image[index, :]
        return mixed_image
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# 举例使用
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.1),
])


import torch

def add_random_noise(img):
    noise = torch.randn_like(img)
    noisy_img = img + noise
    return noisy_img

noisy_transform = transforms.Lambda(lambda x: add_random_noise(x))

train_trans_norm=transforms.Compose([
    transforms.CenterCrop((450, 450)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.1),
    transforms.RandomErasing(),
    MixUpTransform(alpha=1.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_trans_weak = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop((450, 450)),
    transforms.Resize((224, 224)),
    Cutout(),
    transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_trans = transforms.Compose([
    transforms.CenterCrop((450, 450)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_trans_strong = transforms.Compose([transforms.CenterCrop((450, 450)),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             transforms.RandomAutocontrast(),
                                             transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),

                                             transforms.RandomInvert(),

                                             transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                                             transforms.RandomSolarize(random.uniform(0, 1)),
                                             transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2),
                                                                     shear=(-0.3, 0.3, -0.3, 0.3)),
                                             transforms.RandomErasing()
                                             ])


def get_dataloaders():
    NUM_WORKERS=12
    train_dataset1_norm = ISIC2018Dataset(
        csv_file_path='/lables/Train1200.csv',
        img_dir='/data/skin',
            transform=train_trans_norm)
    train_dataset1_strong = ISIC2018Dataset(
        csv_file_path='/Labels/Train1200.csv',
        img_dir='/data/skin',
            transform=train_trans_strong)
    train_dataset1_weak = ISIC2018Dataset(
        csv_file_path='Labels/Train1200.csv',
        img_dir='/data/skin',
            transform=train_trans_weak)

    test_dataset1 = ISIC2018Dataset(
        csv_file_path='Labels/Train1200.csv',
        img_dir='/data/skin',
        transform=test_trans
    )
    train_dataset2_norm = ISIC2018Dataset(
        csv_file_path='Labels/Train_unl1200.csv',
        img_dir='/data/skin',
            transform=train_trans_norm)
    train_dataset2_strong = ISIC2018Dataset(
        csv_file_path='Labels/Train_unl1200.csv',
        img_dir='/data/skin',
            transform=train_trans_strong)
    train_dataset2_weak = ISIC2018Dataset(
        csv_file_path='Labels/Train_unl1200.csv',
        img_dir='/data/skin',
        transform=train_trans_weak)
    test_dataset2 = ISIC2018Dataset(
        csv_file_path='Labels/Test1.csv',
        img_dir='/data/skin',
        transform=test_trans
    )
    train_iter1_norm =DataLoader(train_dataset1_norm,
                       batch_size=4,
                       shuffle=True,
                       drop_last=True,
                       num_workers=NUM_WORKERS)
    train_iter1_strong =DataLoader(train_dataset1_strong,
                       batch_size=4,
                       shuffle=True,
                       drop_last=True,
                       num_workers=NUM_WORKERS)
    train_iter1_weak =DataLoader(train_dataset1_weak,
                       batch_size=4,
                       shuffle=True,
                       drop_last=True,
                       num_workers=NUM_WORKERS)
    test_iter1 = DataLoader(test_dataset1,
                           batch_size=4,
                           num_workers=NUM_WORKERS)
    train_iter2_norm = DataLoader(train_dataset2_norm,
                             batch_size=12,
                             shuffle=True,
                             drop_last=True,
                             num_workers=NUM_WORKERS)
    train_iter2_strong = DataLoader(train_dataset2_strong,
                             batch_size=12,
                             shuffle=True,
                             drop_last=True,
                             num_workers=NUM_WORKERS)
    train_iter2_weak = DataLoader(train_dataset2_weak,
                             batch_size=12,
                             shuffle=True,
                             drop_last=True,
                             num_workers=NUM_WORKERS)
    test_iter2 = DataLoader(test_dataset2,
                            batch_size=12,
                            num_workers=NUM_WORKERS)
    return (train_iter1_norm,train_iter1_weak, train_iter1_strong, test_iter1), (train_iter2_norm,train_iter2_weak, train_iter2_strong,test_iter2)