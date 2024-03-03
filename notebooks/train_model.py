# Standard library imports
import os
import multiprocessing
import gc
import random
import time
import math

# Third-party library imports
# import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import Dict, List

# PyTorch imports
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset

# Local imports
from config_model import ModelConfig, KagglePaths, LocalPaths

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# define the paths
paths = KagglePaths if os.path.exists(KagglePaths.OUTPUT_DIR) else LocalPaths

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 


def get_logger(filename=paths.OUTPUT_DIR):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_spectrogram(spectrogram_path: str):
    """
    Source: https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey
    Visualize spectogram recordings from a parquet file.
    :param spectrogram_path: path to the spectogram parquet.
    """
    sample_spect = pd.read_parquet(spectrogram_path)
    
    split_spect = {
        "LL": sample_spect.filter(regex='^LL', axis=1),
        "RL": sample_spect.filter(regex='^RL', axis=1),
        "RP": sample_spect.filter(regex='^RP', axis=1),
        "LP": sample_spect.filter(regex='^LP', axis=1),
    }
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(np.log(split_spect[split_name]).T, cmap='viridis', aspect='auto', origin='lower')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Log(Value)')
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels([column_name[3:] for column_name in split_spect[split_name].columns])
        frequencies = [column_name[3:] for column_name in split_spect[split_name].columns]
        ax.set_yticks(np.arange(0, len(split_spect[split_name].columns), label_interval))
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()


def get_non_overlap(df_csv, targets):
    # Reference Discussion:
    # https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467021

    # train and validate using only 1 crop per eeg_id
    # same results as Chris's notebook

    tgt_list = targets.tolist()

    agg_dict = {
        'spectrogram_id': 'first',
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'patient_id': 'first',
    }

    for t in tgt_list:
        agg_dict[t] = 'sum'

    agg_dict['expert_consensus'] = 'first'

    train = df_csv.groupby('eeg_id').agg(agg_dict)
    train.columns = ['_'.join(col).strip() for col in train.columns.values]
    train.columns = ['spectrogram_id', 'min', 'max', 'patient_id'] + tgt_list + ['target']
    train = train.reset_index(drop=False)

    train[tgt_list] = train[tgt_list].div(train[tgt_list].sum(axis=1), axis='index')

    return train


class CustomDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        label_cols: List[str],
        config,
        specs: Dict[int, np.ndarray]=None,
        eeg_specs: Dict[int, np.ndarray]=None,
        mode: str = 'train',
    ): 
        self.df = df
        self.label_cols = label_cols
        self.config = config
        self.batch_size = config.BATCH_SIZE
        self.mode = mode
        self.spectograms = specs
        self.eeg_spectograms = eeg_specs
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
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
                        
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
        if self.spectograms:
            for region in range(4):
                img = np.empty((128, 256), dtype='float32')

                spectrogram = self.spectograms[row['spectrogram_id']][r:r+300, region*100:(region+1)*100].T
                spectrogram = self.__transform_spectrogram(spectrogram)
                
                img[14:-14, :] = spectrogram[:, 22:-22] / 2.0
                img_list.append(img)

        if self.eeg_spectograms:
            img = self.eeg_spectograms[row['eeg_id']]
            img_list += [img[:, :, i] for i in range(4)]
        
        # Arrange the data
        if self.data_arrange == 0:
            # --> [512, 512, 1]
            part_1 = np.concatenate(img_list[:4], axis=0)
            part_2 = np.concatenate(img_list[4:], axis=0)
            X = np.concatenate([part_1, part_2], axis=1)
        elif self.data_arrange == 1:
            # --> [256, 256, 4]
            part_1 = np.stack(img_list[:4], axis=2)
            part_2 = np.stack(img_list[4:], axis=2)
            X = np.concatenate([part_1, part_2], axis=0)
        else:
            raise ValueError(f"Data arrangement {self.data_arrange} not supported")
                
        if self.mode == 'train':
            y = row[self.label_cols].values.astype(np.float32)
        elif self.mode == 'test':
            y = np.zeros(len(self.label_cols), dtype=np.float32)

        return X, y

    def __transform_spectrogram(self, spectrogram):

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
    

class CustomModel(nn.Module):

    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()

        self.USE_KAGGLE_SPECTROGRAMS = config.USE_KAGGLE_SPECTROGRAMS
        self.USE_EEG_SPECTROGRAMS = config.USE_EEG_SPECTROGRAMS

        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )
        
        self.preprocess = torch.nn.Conv2d(4, 3, 1, bias=True)
        
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

        if x.shape[-1] == 1:
            x = torch.cat([x, x, x], dim=3)
            x = x.permute(0, 3, 1, 2)
        else:
            x = torch.tensor(x).permute(0, 3, 1, 2)
        
        return x
    
    def forward(self, x):

        x = self.__reshape_input(x)

        if x.shape[1] == 4:
            x = self.preprocess(x)

        x = self.features(x)
        x = self.custom_layers(x)

        return x
    

if __name__ == "__main__":

    # Set the seed
    seed_everything(ModelConfig.SEED)

    # Get the logger
    logger = get_logger()

    # Load the data
    train_csv = pd.read_csv(paths.TRAIN_CSV)
    TARGETS = train_csv.columns[-6:]

    # Get the non-overlapping data
    df_train = get_non_overlap(train_csv, ModelConfig.TARGETS)

    all_specs = np.load(paths.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
    all_eeg_specs = np.load(paths.PRE_LOADED_EEGS, allow_pickle=True).item()

    # Create the dataset
    dataset = CustomDataset(
        df_train,
        TARGETS,
        ModelConfig,
        all_specs,
        all_eeg_specs,
        mode='train'
    )

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=ModelConfig.BATCH_SIZE,
        num_workers=ModelConfig.NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True
    )

    if ModelConfig.VISUALIZE:
        visual_id = np.random.randint(0, len(df_train))
        X, y = dataloader[visual_id]
        print(X.shape, y.shape)

        if X.shape[-1] == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            im = ax.imshow(X[:, :, 0], cmap='viridis')
            plt.colorbar(im, ax=ax)
            fig.suptitle(f"ID={visual_id}; Target: {y}")
            fig.tight_layout()
            plt.show()
        elif X.shape[-1] == 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))
            for i, ax in enumerate(axes.flatten()):
                im = ax[i].imshow(X[:, :, i], cmap='viridis')
                plt.colorbar(im, ax=ax[i])
            fig.suptitle(f"ID={visual_id}; Target: {y}")
            fig.tight_layout()
            plt.show()
        else:
            raise ValueError(f"Data shape {X.shape} not supported")
        
        del X, y

    
