# %% [code]
# Standard library imports
import os
import multiprocessing
import gc
import random
import time

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
from config_hms_models import ModelConfig, KagglePaths, LocalPaths
from my_hms_models import CustomModel, CustomDataset, transform_spectrogram

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

def get_non_overlap(df_csv, targets, group_key=['eeg_id'], calc_method='simple'):
    # Reference Discussion:
    # https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467021

    # train and validate using only 1 crop per eeg_id
    # Simple method:
    # - sum the votes for each label
    # - divide by the total number of votes
    # Weighted method:
    # - calculate the confidence for each label
    # - multiply the confidence by the votes for each label
    # - divide by the sum of the confidence for each label
    # - the confidence is calculated considering:
    #   - the number of votes for the label
    #   - the agreement between the experts

    tgt_list = targets.tolist()
    brain_activity = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    n_classes = len(brain_activity)

    def calc_confidence(row, n_max, n_classes):
        norm_weight = row['total_experts'] / n_max
        agreement = (row['vote_max'] - row['total_experts']/n_classes) / \
                    (row['total_experts'] - row['total_experts']/n_classes)
        return norm_weight * agreement

    def calc_weighted_votes(grp):

        n_experts_max = grp['total_experts'].max()
        grp['confidence'] = grp.apply(calc_confidence, axis=1, args=(n_experts_max, n_classes))
        grp['confidence_norm'] = grp['confidence'] / grp['confidence'].sum()

        weighted_votes = grp[tgt_list].multiply(grp['confidence_norm'], axis='index').sum()

        return weighted_votes

    agg_dict = {
        'spectrogram_id': 'first',
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'patient_id': 'first',
        'expert_consensus': 'first'
    }
    
    groupby = df_csv.groupby(group_key)
    train = groupby.agg(agg_dict)
    train.columns = ['_'.join(col).strip() for col in train.columns.values]
    train.columns = ['spectrogram_id', 'min', 'max', 'patient_id', 'target']

    if calc_method == 'simple':
        vote_sum = groupby[tgt_list].sum()
        class_probs = vote_sum.div(vote_sum.sum(axis=1), axis=0).reset_index(drop=False)

    elif calc_method == 'weighted':
        df_csv['total_experts'] = df_csv[[f"{label}_vote" for label in brain_activity]].sum(axis=1)
        df_csv['vote_max'] = df_csv[[f"{label}_vote" for label in brain_activity]].max(axis=1)
        weighted_votes = df_csv.groupby(group_key).apply(calc_weighted_votes, include_groups=False) 
        class_probs = weighted_votes.div(weighted_votes.sum(axis=1), axis=0).reset_index(drop=False)

    train = train.dropna()
    train = train.reset_index(drop=False)
    train = train.merge(class_probs, on=group_key, how='left')
    
    return train


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


def train_fold(train_loader, valid_loader, config, logger):

    # Create the model
    model = CustomModel(ModelConfig, num_classes=6, pretrained=True)
    model.to(DEVICE)

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

    loss_records = {
        "train": [],
        "valid": []
    }

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, DEVICE, config)
        avg_val_loss, prediction_dict = validate_epoch(valid_loader, model, criterion, DEVICE, config)
    
        elapsed = time.time() - start_time
        info = f"Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s"
        logger.info(info)
        print(f"{'-'*100}\n{info}\n{'-'*100}")

        loss_records['train'].append(avg_train_loss)
        loss_records['valid'].append(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model

    return best_model, prediction_dict, loss_records


def get_result(oof_df, label_cols, target_preds):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result


def train_model(df_train, targets, config, paths, logger):

    # k-fold cross-validation
    gkf = GroupKFold(n_splits=config.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(df_train, df_train['target'], df_train['patient_id'])):
        df_train.loc[valid_index, "fold"] = int(fold)

    print("-"*100)
    print("Validation set sizes")
    print(df_train.groupby("fold").size())
    print("-"*100)
    print(df_train.head(10))

    oof_df = pd.DataFrame()
    for fold in range(config.FOLDS):

        logger.info(f"{'='*100}\nFold: {fold} training\n{'='*100}")
        tik = time.time()

        # ======== SPLIT ==========
        train_folds = df_train[df_train['fold'] != fold].reset_index(drop=True)
        valid_folds = df_train[df_train['fold'] == fold].reset_index(drop=True)
        
        # ======== DATASETS ==========
        train_dataset = CustomDataset(train_folds, targets, config, all_specs, all_eegs, augment=config.AUGMENT, mode="train")
        valid_dataset = CustomDataset(valid_folds, targets, config, all_specs, all_eegs, augment=False, mode="train")
        
        # ======== DATALOADERS ==========
        loader_kwargs = {
            "batch_size": config.BATCH_SIZE,
            "num_workers": config.NUM_WORKERS,
            "pin_memory": True,
            "shuffle": False,
        }
        train_loader = DataLoader(train_dataset, drop_last=True, **loader_kwargs)
        valid_loader = DataLoader(valid_dataset, drop_last=False, **loader_kwargs)

        best_model, prediction_dict, loss_records = train_fold(train_loader, valid_loader, config, logger)

        name_stem = f"{config.MODEL}_fold_{fold}_{config.MODEL_POSTFIX}"
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(loss_records['train'], "o-", label="Train Loss")
        ax.plot(loss_records['valid'], "o-", label="Valid Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"Fold {fold} Training Loss")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f"{paths.OUTPUT_DIR}/{name_stem}_loss.png")

        # Save the model
        save_name = f"{paths.OUTPUT_DIR}/{name_stem}.pth"
        torch.save(best_model.state_dict(), save_name)

        valid_folds[target_preds] = prediction_dict['predictions']
        oof_df = pd.concat([oof_df, valid_folds])

        del train_dataset, valid_dataset, train_loader, valid_loader

        torch.cuda.empty_cache()
        gc.collect()

        result = get_result(valid_folds, TARGETS, target_preds)
        logger.info(f"{'='*100}\nFold {fold} Result: {result} Elapse: {(time.time()-tik) / 60:.2f} min \n{'='*100}")

    return oof_df


if __name__ == "__main__":

    Train_Flag = True

    total_start = time.time()

    target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
    label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
    num_to_label = {v: k for k, v in label_to_num.items()}
    
    torch.cuda.empty_cache()
    gc.collect()

    # Set the seed
    seed_everything(ModelConfig.SEED)

    logger = get_logger(paths.OUTPUT_DIR)
    
    logger.info(f"\nScript Start: {time.ctime()}")
    logger.info(f"Model: {ModelConfig.MODEL}")
    logger.info(f"Model Postfix: {ModelConfig.MODEL_POSTFIX}")
    logger.info(f"Data Arrange: {ModelConfig.DATA_ARRANGE}")
    logger.info(f"Drop Rate: {ModelConfig.DROP_RATE}")
    logger.info(f"Drop Path Rate: {ModelConfig.DROP_PATH_RATE}")
    logger.info(f"Augment: {ModelConfig.AUGMENT}")
    logger.info(f"Non-Overlap Method: {ModelConfig.NON_OVERLAP_METHOD}")
    logger.info(f"Output Dir: {paths.OUTPUT_DIR}")

    # Load the data
    train_csv = pd.read_csv(paths.TRAIN_CSV)
    TARGETS = train_csv.columns[-6:]

    # Get the non-overlapping data
    df_train = get_non_overlap(train_csv, TARGETS, ModelConfig.GROUP_KEYS, calc_method=ModelConfig.NON_OVERLAP_METHOD)

    print(f"{'-'*100}\nTrain Data\n{'-'*100}")
    print(df_train.head(10))
    print(f"{'-'*100}\nTrain Data Shape: {df_train.shape}\n{'-'*100}")

    all_specs = np.load(paths.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
    all_eegs = np.load(paths.PRE_LOADED_EEGS, allow_pickle=True).item()

    if ModelConfig.VISUALIZE:

        # Create the dataset
        dataset = CustomDataset(df_train, TARGETS, ModelConfig, all_specs, all_eegs, mode='train')

        # Create the dataloader
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=ModelConfig.BATCH_SIZE,
            num_workers=ModelConfig.NUM_WORKERS, 
            pin_memory=True, 
            drop_last=True
        )

        visual_id = 108 #np.random.randint(0, len(df_train))
        X, y = dataset[visual_id]
        print(f"Input shape: {X.shape};\nTarget shape: {y.shape}")

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flatten()):
            im = ax.imshow(X[:, :, i], cmap="viridis")
            plt.colorbar(im, ax=ax, orientation='horizontal')
        
        fig.tight_layout()
        fig.savefig(f"{paths.OUTPUT_DIR}/sample_input.png")
        
        del X, y, dataloader, dataset

        torch.cuda.empty_cache()
        gc.collect()

    # ======== K-FOLD Training ==========

    if Train_Flag:
        oof_df = train_model(df_train, TARGETS, ModelConfig, paths, logger)
        oof_df = oof_df.reset_index(drop=True)
        oof_df.to_csv(os.path.join(paths.OUTPUT_DIR, "oof_df.csv"), index=False)
        cv_result = get_result(oof_df, TARGETS, target_preds)
        logger.info(f"{'='*100}\nCV: {cv_result:.8f} Total Running: {(time.time() - total_start) / 60} min\n{'='*100}")
    
    else:
        oof_csv = os.path.join(paths.OUTPUT_DIR, "oof_df.csv")
        if os.path.isfile(oof_csv):    
            oof_df = pd.read_csv(os.path.join(paths.OUTPUT_DIR, "oof_df.csv"))
            cv_result = get_result(oof_df, TARGETS, target_preds)
            logger.info(f"{'='*100}\nCV: {cv_result:.8f} Total Running: {(time.time() - total_start) / 60} min\n{'='*100}")
        else:
            print(f"File Not Found: {oof_csv}")

    