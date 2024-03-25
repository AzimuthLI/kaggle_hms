import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.transforms import v2
import timm
import random
from torch.nn import functional as F
from transformers import ViTMAEModel, ViTMAEConfig, ViTMAEForPreTraining
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


DEFAULT_VITMAE_CONFIG = {
    "architectures": [
        "ViTMAEForPreTraining"
    ],
    "attention_probs_dropout_prob": 0.0,
    "decoder_hidden_size": 512,
    "decoder_intermediate_size": 2048,
    "decoder_num_attention_heads": 16,
    "decoder_num_hidden_layers": 8,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 768,
    "image_size": 224,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "mask_ratio": 0.75,
    "model_type": "vit_mae",
    "norm_pix_loss": False,
    "num_attention_heads": 12,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "patch_size": 16,
    "qkv_bias": True,
    "torch_dtype": "float32",
    "transformers_version": "4.16.0.dev0"
    }


class KagglePaths:
    OUTPUT_DIR = "/kaggle/working/"
    PRE_LOADED_EEGS = '/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTROGRAMS = '/kaggle/input/brain-spectrograms/specs.npy'
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TRAIN_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    TRAIN_SPECTROGRAMS = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
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


class ModelConfig:
    SEED = 20
    SPLIT_ENTROPY = 5.5
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
    DUAL_ENCODER_BACKBONE = 'tf_efficientnet_b2'
    MAE_PRETRAINED_WEIGHTS = 'facebook/vit-mae-base'
    MAE_HIDDEN_DROPOUT_PROB = 0.05
    MAE_ATTENTION_DROPOUT_PROB = 0.05


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


class CustomEfficientNET(nn.Module):

    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(CustomEfficientNET, self).__init__()
        
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
            nn.Linear(self.model.num_features, num_classes)
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


class CustomVITMAE(nn.Module):
    def __init__(self, config, num_classes=6, pretrained=None):
        super(CustomVITMAE, self).__init__()

        # Load the ViTMAE configuration and model as before
        mae_config = ViTMAEConfig.from_dict(DEFAULT_VITMAE_CONFIG)
        mae_config.hidden_dropout_prob = config.MAE_HIDDEN_DROPOUT_PROB
        mae_config.attention_probs_dropout_prob = config.MAE_ATTENTION_DROPOUT_PROB

        if pretrained:
            print(f"Loading pretrained weights from {config.MAE_PRETRAINED_WEIGHTS}")
            self.vitmae = ViTMAEModel.from_pretrained(
                config.MAE_PRETRAINED_WEIGHTS, config=mae_config)
        else:
            self.vitmae = ViTMAEModel(config=mae_config)

        # Assumes the [CLS] token (or equivalent) is at position 0 
        # and directly usable for classification
        # self.classifier = nn.Linear(self.vitmae.config.hidden_size, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.vitmae.config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(config.DROP_RATE),
            nn.Linear(512, num_classes)
        )

    def __reshape_input(self, x): # <- (N, C=8, H=128, W=256)

        concat_p1 = torch.cat(torch.chunk(x[:, :4, :, :], 4, dim=1), dim=2)
        concat_p2 = torch.cat(torch.chunk(x[:, 4:, :, :], 4, dim=1), dim=2)
        x_concat = torch.cat((concat_p1, concat_p2), dim=3)
       
        resized = F.interpolate(x_concat, size=(224, 224), mode='bilinear', align_corners=False)
        stacked = resized.repeat(1, 3, 1, 1)
        
        return stacked #-> (N, C=3, H=224, W=224)
    
    def forward(self, x):
        x = self.__reshape_input(x)
        outputs = self.vitmae(pixel_values=x)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])

        return logits
    

class DualEncoderModel(nn.Module):
    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(DualEncoderModel, self).__init__()

        backbone = config.DUAL_ENCODER_BACKBONE

        self.eeg_model = timm.create_model(
            backbone,
            pretrained=pretrained,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )

        self.spec_model = timm.create_model(
            backbone,
            pretrained=pretrained,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )
        
        if config.FREEZE:
            for i,(name, param) in enumerate(list(self.model.named_parameters())\
                                            [0:config.NUM_FROZEN_LAYERS]):
                param.requires_grad = False

        self.eeg_features = nn.Sequential(*list(self.eeg_model.children())[:-2])
        self.spec_features = nn.Sequential(*list(self.spec_model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.eeg_model.num_features, num_classes)
        )

    def __reshape_input(self, x):
        # # raw implementation
        # # input size: [batch * 128 * 256 * 8]
        # ## --> 256*512*3 for 2 parts
        # spectograms = torch.cat([x[:, :, :, i:i+1] for i in range(4)], dim=1) 
        # eegs = torch.cat([x[:, :, :, i:i+1] for i in range(4,8)], dim=1)
        # spec = torch.cat([spectograms, spectograms, spectograms], dim=3)
        # spec = spec.permute(0, 3, 1, 2)
        # eeg = torch.cat([eegs, eegs, eegs], dim=3)
        # eeg = eeg.permute(0, 3, 1, 2)

        # new implementation
        # input size: [N, C=8, H=128, W=256]
        specs = torch.cat([x[:, i:i+1, :, :] for i in range(4)], dim=2) #-> [N, 1, 512, 256]
        specs = torch.cat([specs]*3, dim=1) #-> [N, 3, 512, 256]

        eegs = torch.cat([x[:, i:i+1, :, :] for i in range(4, 8)], dim=2) #-> [N, 1, 512, 256]
        eegs = torch.cat([eegs]*3, dim=1) #-> [N, 3, 512, 256]
        
        return eegs, specs
    
    def forward(self, x):
        eeg, spec = self.__reshape_input(x)
        eeg_feature = self.eeg_features(eeg)
        spec_feature = self.spec_features(spec)
        x = self.custom_layers(eeg_feature + spec_feature)
        return x


# MAE Pretraining and LightGBM functions
class PreTrainDataset(Dataset):

    def __init__(
        self, 
        df: pd.DataFrame,
        all_specs: Dict[str, np.ndarray],
        all_eegs: Dict[str, np.ndarray],
        mode: str = 'train',
    ): 
        self.df = df
        self.spectrograms = all_specs
        self.eeg_spectrograms = all_eegs
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        X = self.__data_generation(index)
        X = self.__transform(X)
        return X
    
    def __data_generation(self, index): # --> [(C=8) x (H=128) x (W=256)]
        
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
      
        X = np.array(img_list, dtype='float32')
        X = torch.tensor(X, dtype=torch.float32)
        
        return X

    def __transform(self, x):
        # To be implemented...
        return x 


def reshape_pretrain_input(x): #<- (N, C, H, W)
    x = torch.stack(x, dim=0)
    concat_p1 = torch.cat(torch.chunk(x[:, :4, :, :], 4, dim=1), dim=2)
    concat_p2 = torch.cat(torch.chunk(x[:, 4:, :, :], 4, dim=1), dim=2)
    x_concat = torch.cat((concat_p1, concat_p2), dim=3)
   
    resized = F.interpolate(x_concat, size=(224, 224), mode='bilinear', align_corners=False)
    stacked = resized.repeat(1, 3, 1, 1)
    
    return stacked


def generate_pretrain_features(df, all_specs, all_eegs, pretrained_path, device, mode='train'):

    dataset = PreTrainDataset(df, all_specs, all_eegs, mode=mode)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=reshape_pretrain_input)

    mae_config = ViTMAEConfig.from_dict(DEFAULT_VITMAE_CONFIG)

    ft_extractor = ViTMAEForPreTraining.from_pretrained(pretrained_path, config=mae_config)
    ft_extractor = ft_extractor.to(device)

    ft_extractor.eval()

    feature_collects = []
    for x in tqdm(dataloader):
        x = x.to(device)
        with torch.no_grad():
            output = ft_extractor(x, output_hidden_states=True)
            features = output.hidden_states[-1][:, 0, :]
            feature_collects.append(features.cpu().numpy())

    feature_matrix = np.concatenate(feature_collects, axis=0)

    return feature_matrix


