
import numpy as np
import pandas as pd
from typing import List, Dict

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.transforms import v2
import timm
import random


class MyXYMasking(nn.Module):

    def __init__(self, mask_ratio=0.1, max_mask_num=2):
        super(MyXYMasking, self).__init__()
        self.mask_ratio = mask_ratio
        self.max_mask_num = max_mask_num
    
    def forward(self, image):
    
        # Assuming image is a tensor of shape [H, W, C]
        h, w, _ = image.shape
        
        for _ in range(random.randint(1, self.max_mask_num)):
            image = self._apply_mask(image, h, w)

        return image

    def _apply_mask(self, image, h, w):
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


applier = v2.RandomApply(
    transforms=[
        v2.RandomHorizontalFlip(p=.2),
        v2.RandomVerticalFlip(p=.2),
        MyXYMasking(mask_ratio=0.1, max_mask_num=2),
        ], 
        p=.5)


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

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.mode == 'train' and self.augment:
            X = self.__transform(X)
        
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

        return X, y

    def __transform(self, x):
        x = applier(x)
        return x

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
        # input size: [batch * 128 * 256 * 8]
        # output size: [batch * 3 * 512 * 512]
        spectograms = torch.cat([x[:, :, :, i:i+1] for i in range(4)], dim=1) 
        eegs = torch.cat([x[:, :, :, i:i+1] for i in range(4,8)], dim=1)
        x = torch.cat([spectograms, eegs], dim=2)
        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        
        return x
    
    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x
