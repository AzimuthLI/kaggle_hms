import os
import numpy as np
import pandas as pd
from typing import List, Dict

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.transforms import v2
import timm
import random


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
    PRE_LOADED_SPECTROGRAMS = './inputs/brain-spectrograms/specs.npy'
    TRAIN_CSV = "./inputs/hms-harmful-brain-activity-classification/train.csv"
    TRAIN_EEGS = "./inputs/hms-harmful-brain-activity-classification/train_eegs"
    TRAIN_SPECTROGRAMS = "./inputs/hms-harmful-brain-activity-classification/train_spectrograms"
    TEST_CSV = "./inputs/hms-harmful-brain-activity-classification/test.csv"
    TEST_SPECTROGRAMS = "./inputs/hms-harmful-brain-activity-classification/test_spectrograms"
    TEST_EEGS = "./inputs/hms-harmful-brain-activity-classification/test_eegs"


class JobConfig:
    SEED = 20
    OUTPUT_DIR = KagglePaths.OUTPUT_DIR if os.path.exists(KagglePaths.OUTPUT_DIR) else LocalPaths.OUTPUT_DIR
    PATHS = KagglePaths if os.path.exists(KagglePaths.OUTPUT_DIR) else LocalPaths
    K_FOLD = 5
    ENTROPY_SPLIT = 5.5 
    VISUALIZE = True


class ModelConfig:
    MODEL_NAME = "EfficientNet_b2_two_stage"
    MODEL_BACKBONE = "tf_efficientnet_b2"
    BATCH_SIZE = 16
    EPOCHS = 6
    GRADIENT_ACCUMULATION_STEPS = 2
    DROP_RATE = 0.15 # default: 0.1
    DROP_PATH_RATE = 0.25 # default: 0.2
    WEIGHT_DECAY = 0.01
    REGULARIZATION = None
    DROP_RATE = 0.15 # default: 0.1
    DROP_PATH_RATE = 0.25 # default: 0.2
    USE_KAGGLE_SPECTROGRAMS = True
    USE_EEG_SPECTROGRAMS = True
    AMP = True
    AUGMENT = False
    AUGMENTATIONS = ['h_flip', 'v_flip', 'xy_masking', 'cutmix']
    PRINT_FREQ = 50
    FREEZE = False
    NUM_FROZEN_LAYERS = 0
    NUM_WORKERS = 0 
    MAX_GRAD_NORM = 1e7


# AUGMENTATIONS
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


class CustomDataset(Dataset):

    def __init__(
        self, 
        df: pd.DataFrame,
        label_cols: List[str],
        config,
        all_specs: Dict[str, np.ndarray],
        all_eegs: Dict[str, np.ndarray],
        mode: str = 'train',
    ): 
        self.df = df
        self.label_cols = label_cols
        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.mode = mode
        self.spectrograms = all_specs
        self.eeg_spectrograms = all_eegs

        if self.config.AUGMENT:
            self.augment_applier, self.cutmix_applier = self.__get_augment_applier()
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        
        X, y = self.__data_generation(index)

        if self.mode == 'train' and self.config.AUGMENT:
            X, y = self.__transform(X, y)
        
        return X, y
    
    def __data_generation(self, index): # --> [(C=8) x (H=128) x (W=256)]
        
        row = self.df.iloc[index]
        if self.mode=='test': 
            r = 0
        else: 
            r = int((row['min'] + row['max']) // 4)
        
        img_list = []
        if self.config.USE_KAGGLE_SPECTROGRAMS:
            for region in range(4):
                img = np.zeros((128, 256), dtype='float32')

                spectrogram = self.spectrograms[row['spectrogram_id']][r:r+300, region*100:(region+1)*100].T
                spectrogram = transform_spectrogram(spectrogram)
                
                img[14:-14, :] = spectrogram[:, 22:-22] / 2.0
                img_list.append(img)

        if self.config.USE_EEG_SPECTROGRAMS:
            img = self.eeg_spectrograms[row['eeg_id']]
            img_list += [img[:, :, i] for i in range(4)]
      
        X = np.array(img_list, dtype='float32')
        X = torch.tensor(X, dtype=torch.float32)
                
        if (self.mode == 'train') or (self.mode == 'valid'):
            y = row[self.label_cols].values.astype(np.float32)
        elif self.mode == 'test':
            y = np.zeros(len(self.label_cols), dtype=np.float32)
        else:
            raise ValueError(f"Invalid mode {self.mode}!")
        
        y = torch.tensor(y, dtype=torch.float32)
        
        return X, y
    
    def __get_augment_applier(self):

        aug_list = []
        
        if 'h_flip' in self.config.AUGMENTATIONS:
            aug_list.append(v2.RandomHorizontalFlip(p=1))
        if 'v_flip' in self.config.AUGMENTATIONS:
            aug_list.append(v2.RandomVerticalFlip(p=1))
        if 'xy_masking' in self.config.AUGMENTATIONS:
            aug_list.append(MyXYMasking(mask_ratio=0.1, max_mask_num=1, p=1))

        if len(aug_list) > 0:
            augment_applier = v2.RandomApply(transforms=aug_list, p=.5)
        else:
            augment_applier = None

        cutmix_applier = None
        if 'cutmix' in self.config.AUGMENTATIONS:
            cutmix_applier = MyCutMix(cut_ratio=0.2, bound_pad=20, p=1)
        
        return augment_applier, cutmix_applier

    def __transform(self, x, y):

        if random.choice([True, False]):
            if self.augment_applier is not None:
                x = self.augment_applier(x)
        else:
            if self.cutmix_applier is not None:
                sample_class = self.label_cols[torch.argmax(y).item()]
                same_class_samples = self.df[self.df[self.label_cols].idxmax(axis=1) == sample_class]
                # Skip if less than 2 samples with the same class label
                if len(same_class_samples) > 2:
                    selected_idx = random.choice(same_class_samples.index)
                    cut_img, cut_target = self.__data_generation(selected_idx)
                    x, y = self.cutmix_applier(x, y, cut_img, cut_target)

        return x, y


class CustomModel(nn.Module):

    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()
        
        self.config = config

        self.model = timm.create_model(
            config.MODEL_BACKBONE,
            pretrained=pretrained,
            drop_rate = config.DROP_RATE,
            drop_path_rate = config.DROP_PATH_RATE,
        )

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
            nn.Softmax()
        )
    
    def __reshape_input(self, x): # <- [(C=8) x (H=128) x (W=256)]
        
        if self.config.USE_KAGGLE_SPECTROGRAMS:
            kgg_specs = torch.cat([x[:, i:i+1, :, :] for i in range(0, 4)], dim=2)
            
        if self.config.USE_EEG_SPECTROGRAMS:
            eeg_specs = torch.cat([x[:, i:i+1, :, :] for i in range(4, 8)], dim=2)
            
        if self.config.USE_KAGGLE_SPECTROGRAMS and self.config.USE_EEG_SPECTROGRAMS:
            x = torch.cat([kgg_specs, eeg_specs], dim=3)
        else:
            x = kgg_specs if self.config.USE_KAGGLE_SPECTROGRAMS else eeg_specs

        x = torch.cat([x, x, x], dim=1)
        
        return x # ->[(C=3) x (H=512) x (W=512)]
    
    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x
    


# class CustomVITMAE()