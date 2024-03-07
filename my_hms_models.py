# %% [code]

import numpy as np
import pandas as pd
from typing import List, Dict

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.transforms import v2
import timm
import random

class ModelConfig:
    NON_OVERLAP_METHOD = 'weighted' # 'weighted', 'simple'
    GROUP_KEYS = 'eeg_id'
    AMP = True
    BATCH_SIZE = 16
    EPOCHS = 6
    FOLDS = 5
    GRADIENT_ACCUMULATION_STEPS = 2
    DROP_RATE = 0.15 # default: 0.1
    DROP_PATH_RATE = 0.25 # default: 0.2
    USE_KAGGLE_SPECTROGRAMS = True
    USE_EEG_SPECTROGRAMS = True
    AUGMENT = True
    DATA_ARRANGE = 0 # 0: [512, 512, 1], 1: [256, 256, 4]
    FREEZE = False
    MAX_GRAD_NORM = 1e7
    MODEL = "tf_efficientnet_b2" #"tf_efficientnet_b0"
    MODEL_POSTFIX = "h_flip" # "flat"
    NUM_FROZEN_LAYERS = 39
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 50
    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = True
    WEIGHT_DECAY = 0.01
    
class KagglePaths:
    OUTPUT_DIR = "/kaggle/working/"
    PRE_LOADED_EEGS = '/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTOGRAMS = '/kaggle/input/brain-spectrograms/specs.npy'
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TRAIN_EEGS = "/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms/"
    TRAIN_SPECTOGRAMS = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_SPECTROGRAMS = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

class LocalPaths:
    OUTPUT_DIR = "./outputs/"
    PRE_LOADED_EEGS = './inputs/brain-eeg-spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTOGRAMS = './inputs/brain-spectrograms/specs.npy'
    TRAIN_CSV = "./inputs/hms-harmful-brain-activity-classification/train.csv"
    TRAIN_EEGS = " ./inputs/hms-harmful-brain-activity-classification/train_eegs"
    TRAIN_SPECTOGRAMS = "./inputs/hms-harmful-brain-activity-classification/train_spectrograms"
    TEST_CSV = "./inputs/hms-harmful-brain-activity-classification/test.csv"
    TEST_SPECTROGRAMS = "./inputs/hms-harmful-brain-activity-classification/test_spectrograms"
    TEST_EEGS = "./inputs/hms-harmful-brain-activity-classification/test_eegs"


def transform_spectrogram(spectrogram):

    # Log transform spectogram
    spectrogram = np.clip(spectrogram, np.exp(-4), np.exp(8))
    spectrogram = np.log(spectrogram)

    # Standarize per image
    ep = 1e-6
    mu = np.nanmean(spectrogram.flatten())
    std = np.nanstd(spectrogram.flatten())
    spectrogram = (spectrogram-mu) / (std+ep)
    spectrogram = np.nan_to_num(spectrogram, nan=0.0)
        
    return spectrogram


class MyXYMasking(nn.Module):

    def __init__(self, mask_ratio=0.1, max_mask_num=2, p=0.5):
        super(MyXYMasking, self).__init__()
        self.mask_ratio = mask_ratio
        self.max_mask_num = max_mask_num
        self.p = p
    
    def forward(self, X):
        if random.random() > self.p:
            return X
        else:
            for _ in range(random.randint(1, self.max_mask_num)):
                X = self._apply_mask(X)
        return X

    def _apply_mask(self, image):

        c, h, w = image.shape
        # Calculate max mask width and height based on mask_ratio
        max_mask_width = int(w * self.mask_ratio)
        max_mask_height = int(h * self.mask_ratio)
        
        # Randomly choose the dimension(s) to mask: 0 for cols, 1 for rows, 2 for both
        dim_to_mask = random.randint(0, 2)
        
        # Initialize mask as ones
        mask = torch.ones_like(image)
        
        # Select random start point for masking in both dimensions
        x1 = random.randint(0, w - max_mask_width)
        y1 = random.randint(0, h - max_mask_height)
        
        # Calculate end points ensuring they do not exceed max mask width/height
        x2 = x1 + random.randint(1, max_mask_width)
        y2 = y1 + random.randint(1, max_mask_height)
        
        # Adjust x2 and y2 to not go beyond image bounds
        x2 = min(x2, w)
        y2 = min(y2, h)
        
        if dim_to_mask == 0 or dim_to_mask == 2:
            mask[:, :, x1:x2] = 0
        
        if dim_to_mask == 1 or dim_to_mask == 2:
            mask[:, y1:y2, :] = 0
        
        image_new = image * mask
        
        return image_new


class MyCutMix(nn.Module):

    def __init__(self, cut_ratio=0.1, bound_pad=20, p=0.5):
        super(MyCutMix, self).__init__()
        self.cut_ratio = cut_ratio
        self.bound_pad = bound_pad
        self.p = p

    def forward(self, X, y, cut_image, cut_target):

        if random.random() > self.p:
            return X, y
        else:
            # Randomly select a region to cut
            # only apply cut along the time axis
            # avoid cutting the center of the spectrogram
            c, h, w = X.shape
            center = int(w/2)
            cut_width = random.randint(1, int(w * self.cut_ratio))

            # Either cut the left or right side
            if random.choice([True, False]):
                x0 = random.randint(self.bound_pad, center - cut_width - self.bound_pad)
                x1 = x0 + cut_width
            else:
                x0 = random.randint(center + self.bound_pad, w - cut_width - self.bound_pad)
                x1 = x0 + cut_width

            # Apply the cut
            X[:, :, x0:x1] = cut_image[:, :, x0:x1]

            # Adjust target
            cut_area = (x1 - x0) / w
            y = (1-cut_area) * y + cut_area * cut_target

            return X, y



# Define the augmentations
augment_applier = v2.RandomApply(
    transforms=[
        v2.RandomHorizontalFlip(p=1), 
        # v2.RandomVerticalFlip(p=1),
        # MyXYMasking(mask_ratio=0.1, max_mask_num=1, p=1),
        ], 
        p=.5)

# maximum cut: 20% of the width
cutmix_applier = MyCutMix(cut_ratio=0.2, bound_pad=20, p=.5)


class CustomDataset(Dataset):

    def __init__(
        self, 
        df: pd.DataFrame,
        label_cols: List[str],
        config,
        all_specs: Dict[str, np.ndarray],
        all_eegs: Dict[str, np.ndarray],
        augment: bool = False,
        mode: str = 'train',
    ): 
        self.df = df
        self.label_cols = label_cols
        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.mode = mode
        self.spectrograms = all_specs
        self.eeg_spectrograms = all_eegs
        self.augment = augment
        self.data_arrange = config.DATA_ARRANGE
        
    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)
        
    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)

        if self.mode == 'train' and self.augment:
            X, y = self.__transform(X, y)
        
        return X, y
                        
    def __data_generation(self, index):
        """
        Generates data containing batch_size samples.
        """
        row = self.df.iloc[index]
        if self.mode=='test': 
            r = 0
        else: 
            r = int((row['min'] + row['max']) // 4)
        
        img_list = []

        for region in range(4):
            img = np.zeros((128, 256), dtype='float32')

            spectrogram = self.spectrograms[row['spectrogram_id']][r:r+300, region*100:(region+1)*100].T
            spectrogram = transform_spectrogram(spectrogram)
            
            img[14:-14, :] = spectrogram[:, 22:-22] / 2.0
            img_list.append(img)

        img = self.eeg_spectrograms[row['eeg_id']]
        img_list += [img[:, :, i] for i in range(4)]

        X = np.stack(img_list, axis=2)
                
        if self.mode == 'train':
            y = row[self.label_cols].values.astype(np.float32)
        elif self.mode == 'test':
            y = np.zeros(len(self.label_cols), dtype=np.float32)

        X = torch.tensor(X, dtype=torch.float32)
        X = X.permute(2, 0, 1)

        y = torch.tensor(y, dtype=torch.float32)
        
        return X, y

    def __transform(self, x, y):

        if True: #random.choice([True, False]):
            x = augment_applier(x)
        else:
            
            sample_class = self.label_cols[torch.argmax(y).item()]
            same_class_samples = self.df[self.df[self.label_cols].idxmax(axis=1) == sample_class]

            # Skip if less than 2 samples with coconfidence score 1.0
            if len(same_class_samples) > 2:

                selected_idx = random.choice(same_class_samples.index)
                cut_img, cut_target = self.__data_generation(selected_idx)

                x, y = cutmix_applier(x, y, cut_img, cut_target)

        return x, y


class CustomModel(nn.Module):

    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()

        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate = config.DROP_RATE,
            drop_path_rate = config.DROP_PATH_RATE,
        )
        
        if config.FREEZE:
            for i,(name, param) in enumerate(list(self.model.named_parameters())\
                                             [0:config.NUM_FROZEN_LAYERS]):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes)
        )
    
    def __reshape_input(self, x):
        # # input size: [batch * 128 * 256 * 8]
        # spectograms = torch.cat([x[:, :, :, i:i+1] for i in range(4)], dim=1) 
        # eegs = torch.cat([x[:, :, :, i:i+1] for i in range(4,8)], dim=1)
        # x = torch.cat([spectograms, eegs], dim=2)
        # x = torch.cat([x, x, x], dim=3)
        # x = x.permute(0, 3, 1, 2)

        # input size: [batch * 8 * 128 * 256]
        kgg_specs = torch.cat([x[:, i:i+1, :, :] for i in range(0, 4)], dim=2)
        eeg_specs = torch.cat([x[:, i:i+1, :, :] for i in range(4, 8)], dim=2)

        x = torch.cat([kgg_specs, eeg_specs], dim=3)
        x = torch.cat([x, x, x], dim=1)
        
        # output size: [batch * 3 * 512 * 512]
        return x
    
    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x