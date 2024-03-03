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
from sklearn.model_selection import KFold, GroupKFold

# PyTorch imports
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

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


def get_logger(log_dir):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    filename = os.path.join(log_dir, "train.log")
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=filename, mode="a+")
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
                img = np.zeros((128, 256), dtype='float32')

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
            X = np.expand_dims(X, axis=2)
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


def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device, config):

    model.train()
    scaler = GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    global_step = 0
    start = end = time.time()

    with tqdm(train_loader, unit="train_batch", desc='Train') as tk0:
        
        for step, (X, y) in enumerate(tk0):
            X, y = X.to(device), y.to(device)
            
            batch_size = y.size(0)

            with autocast(enabled=config.AMP):
                outputs = model(X)
                loss = criterion(F.log_softmax(outputs, dim=1), y)

            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            scheduler.step()

            end = time.time()

            if step % config.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                lr = scheduler.get_last_lr()[0]
                info = f"Epoch: [{epoch+1}][{step}/{len(train_loader)}]"
                info += f"Elapsed {(end-start):.2f}s Loss: {losses.avg:.4f} Grad: {grad_norm:.4f} LR: {lr:.8f}"
                print(info)

    return losses.avg


def validate_epoch(valid_loader, model, criterion, device, config):

    model.eval()
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    start = end = time.time()

    preds = []

    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tk0:
        for step, (X, y) in enumerate(tk0):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)

            with torch.no_grad():
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)

            losses.update(loss.item(), batch_size)
            preds.append(y_preds.to('cpu').numpy())
            end = time.time()

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(valid_loader)-1):
                info = f"EVAL: [{step}/{len(valid_loader)}] Elapsed {(end - start):.2f}s Loss: {losses.avg:.4f}"
                print(info)

    prediction_dict = {"predictions": np.concatenate(preds)}

    return losses.avg, prediction_dict


def train_loop(model, train_loader, valid_loader, config):

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    criterion = nn.KLDivLoss(reduction="batchmean")
    best_loss = np.inf

    best_model = None

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, DEVICE, config)
        avg_val_loss, prediction_dict = validate_epoch(valid_loader, model, criterion, DEVICE, config)
    
        elapsed = time.time() - start_time
        info = f"Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s"
        print(f"{'-'*100}\n{info}\n{'-'*100}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model

    return best_model, prediction_dict


def get_result(oof_df, label_cols, target_preds):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result


if __name__ == "__main__":

    target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
    label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
    num_to_label = {v: k for k, v in label_to_num.items()}

    # Set the seed
    seed_everything(ModelConfig.SEED)

    logger = get_logger(paths.OUTPUT_DIR)

    # Load the data
    train_csv = pd.read_csv(paths.TRAIN_CSV)
    TARGETS = train_csv.columns[-6:]

    # Get the non-overlapping data
    df_train = get_non_overlap(train_csv, TARGETS)

    all_specs = np.load(paths.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
    all_eeg_specs = np.load(paths.PRE_LOADED_EEGS, allow_pickle=True).item()

    if ModelConfig.VISUALIZE:

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

        visual_id = np.random.randint(0, len(df_train))
        X, y = dataset[visual_id]
        print(f"Input shape: {X.shape};\nTarget shape: {y.shape}")

        if X.shape[-1] == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            im = ax.imshow(X[:, :, 0], cmap='viridis')
            plt.colorbar(im, ax=ax)
            fig.suptitle(f"ID={visual_id}; Target: {y}")
            fig.tight_layout()
            fig.savefig(f"{paths.OUTPUT_DIR}/sample_spectrogram.png")
        elif X.shape[-1] == 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))
            for i, ax in enumerate(axes.flatten()):
                im = ax[i].imshow(X[:, :, i], cmap='viridis')
                plt.colorbar(im, ax=ax[i])
            fig.suptitle(f"ID={visual_id}; Target: {y}")
            fig.tight_layout()
            fig.savefig(f"{paths.OUTPUT_DIR}/sample_spectrogram.png")
        else:
            raise ValueError(f"Data shape {X.shape} not supported")
        
        del X, y, dataloader, dataset


    # Create the model
    model = CustomModel(ModelConfig, num_classes=6, pretrained=True)
    model.to(DEVICE)

    # k-fold cross-validation
    gkf = GroupKFold(n_splits=ModelConfig.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(df_train, df_train['target'], df_train['patient_id'])):
        df_train.loc[valid_index, "fold"] = int(fold)

    print("-"*100)
    print("Validation set sizes")
    print(df_train.groupby("fold").size())
    print("-"*100)
    print(df_train.head(10))

    oof_df = pd.DataFrame()
    for fold in range(ModelConfig.FOLDS):

        logger.info(f"{'='*100}\nFold: {fold} training\n{'='*100}")
        tik = time.time()

        # ======== SPLIT ==========
        train_folds = df_train[df_train['fold'] != fold].reset_index(drop=True)
        valid_folds = df_train[df_train['fold'] == fold].reset_index(drop=True)
        
        # ======== DATASETS ==========
        train_dataset = CustomDataset(train_folds, TARGETS, ModelConfig, all_specs, all_eeg_specs, mode="train")
        valid_dataset = CustomDataset(valid_folds, TARGETS, ModelConfig, all_specs, all_eeg_specs, mode="train")
        
        # ======== DATALOADERS ==========
        loader_kwargs = {
            "batch_size": ModelConfig.BATCH_SIZE,
            "num_workers": ModelConfig.NUM_WORKERS,
            "pin_memory": True,
            "shuffle": False,
        }
        train_loader = DataLoader(train_dataset, drop_last=True, **loader_kwargs)
        valid_loader = DataLoader(valid_dataset, drop_last=False, **loader_kwargs)

        best_model, prediction_dict = train_loop(model, train_loader, valid_loader, ModelConfig)

        # Save the model
        save_name = f"{paths.OUTPUT_DIR}/{ModelConfig.MODEL}_fold_{fold}_{ModelConfig.MODEL_POSTFIX}.pth"
        torch.save(best_model.state_dict(), save_name)

        valid_folds[target_preds] = prediction_dict['predictions']
        oof_df = pd.concat([oof_df, valid_folds])

        print(valid_folds.head())

        del train_dataset, valid_dataset, train_loader, valid_loader

        torch.cuda.empty_cache()
        gc.collect()

        result = get_result(valid_folds, TARGETS, target_preds)
        logger.info(f"{'='*100}\nFold {fold} Result: {result} Elapse: {(time.time()-tik) / 60:.2f} min \n{'='*100}")

    oof_df = oof_df.reset_index(drop=True)
    cv_result = get_result(oof_df, TARGETS, target_preds)
    logger.info(f"{'='*100}\nCV: {cv_result}\n{'='*100}")
    oof_df.to_csv(os.path.join(paths.OUTPUT_DIR, "oof_df.csv"), index=False)