from tqdm.notebook import tqdm
import logging
import random
import numpy as np
import pandas as pd
import os, gc
from time import time, ctime
import warnings

from scipy.special import rel_entr, softmax
from sklearn.model_selection import GroupKFold, KFold
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from kl_divergence import score as kaggle_score
from engine_hms_model import CustomDataset, CustomEfficientNET, CustomVITMAE, DualEncoderModel


# CONSTANTS
BRAIN_ACTIVITY = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
TARGETS = [f"{lb}_vote" for lb in BRAIN_ACTIVITY]
TARGETS_PRED = [f"{lb}_pred" for lb in BRAIN_ACTIVITY]


def get_logger(log_dir, logger_name="train_model.log"):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger_file = os.path.join(log_dir, logger_name)
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=logger_file, mode="a+")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


# FUNCTIONS
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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


def calc_entropy(row, tgt_list):
    nc = len(tgt_list)
    uniform_list = [1 / nc for i in range(nc)]
    return sum(rel_entr(uniform_list, row[tgt_list].astype('float32').values + 1e-5))


def gen_non_overlap_samples(df_csv, targets):
    # Reference Discussion:
    # https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467021

    tgt_list = targets.tolist()
    brain_activity = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']

    agg_dict = {
        'spectrogram_id': 'first',
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'patient_id': 'first',
        'expert_consensus': 'first'
    }

    for tgt in tgt_list:
        agg_dict[tgt] = 'sum'

    # groupby = df_csv.groupby(['eeg_id'] + tgt_list)
    groupby = df_csv.groupby(['eeg_id'])
    train = groupby.agg(agg_dict)
    train = train.reset_index()
    # train.columns = ['_'.join(col).strip() for col in train.columns.values]
    # train.columns = ["eeg_id"] + tgt_list + ['spectrogram_id', 'min', 'max', 'patient_id', 'target']
    train.columns = ['eeg_id', 'spectrogram_id', 'min', 'max', 'patient_id', 'target'] + tgt_list
    train['total_votes'] = train[tgt_list].sum(axis=1)
    train[tgt_list] = train[tgt_list].apply(lambda x: x / x.sum(), axis=1)
    
    # vote_sum = train[tgt_list]
    # train[tgt_list] = vote_sum.div(vote_sum.sum(axis=1), axis=0)
    
    return train


def load_kaggle_data(train_csv, preload_specs, preload_eegs, split_entropy=5.5):

    train_csv = pd.read_csv(train_csv)
    targets = train_csv.columns[-6:]

    train_csv['entropy'] = train_csv.apply(calc_entropy, axis=1, tgt_list=targets)
    train_csv['total_votes'] = train_csv[targets].sum(axis=1)

    easy_csv = train_csv[train_csv['entropy'] >= split_entropy].copy().reset_index(drop=True)
    hard_csv = train_csv[train_csv['entropy'] < split_entropy].copy().reset_index(drop=True)

    train_easy = gen_non_overlap_samples(easy_csv, targets)
    train_hard = gen_non_overlap_samples(hard_csv, targets)

    all_specs = np.load(preload_specs, allow_pickle=True).item()
    all_eegs = np.load(preload_eegs, allow_pickle=True).item()

    return train_easy, train_hard, all_specs, all_eegs


class Trainer:

    def __init__(self, model, config, logger):

        self.model = model
        self.logger = logger
        self.config = config
        
        self.early_stop_rounds = config.EARLY_STOP_ROUNDS
        self.early_stop_counter = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def train(self, train_loader, valid_loader, from_checkpoint=None):

        self.optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=self.config.WEIGHT_DECAY)

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=1e-4,
            epochs=self.config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=100,
        )

        if from_checkpoint is not None:
            self.model.load_state_dict(torch.load(from_checkpoint, map_location=self.device))

        self.model.to(self.device)
        best_weights, best_preds, best_loss = None, None, float("inf")
        loss_records = {"train": [], "valid": []}

        for epoch in range(self.config.EPOCHS):
            start_epoch = time()

            train_loss, _ = self._train_or_valid_epoch(epoch, train_loader, is_train=True)
            valid_loss, valid_preds = self._train_or_valid_epoch(epoch, valid_loader, is_train=False)

            loss_records["train"].append(train_loss)
            loss_records["valid"].append(valid_loss)

            elapsed = time() - start_epoch

            info = f"{'-' * 100}\nEpoch {epoch + 1} - "
            info += f"Average Loss: (train) {train_loss:.4f}; (valid) {valid_loss:.4f} | Time: {elapsed:.2f}s"
            self.logger.info(info)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_weights = self.model.state_dict()
                best_preds = valid_preds
                self.logger.info(f"Best model found in epoch {epoch + 1} | valid loss: {best_loss:.4f}")
                self.early_stop_counter = 0
            
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_rounds:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return best_weights, best_preds, loss_records

    def _train_or_valid_epoch(self, epoch_id, dataloader, is_train=True):

        self.model.train() if is_train else self.model.eval()
        mode = "Train" if is_train else "Valid"

        len_loader = len(dataloader)
        scaler = GradScaler(enabled=self.config.AMP)
        loss_meter, predicts_record = AverageMeter(), []

        start = time()
        pbar = tqdm(dataloader, total=len(dataloader), unit="batch", desc=f"{mode} [{epoch_id}]")
        for step, (X, y) in enumerate(pbar):
            X, y = X.to(self.device), y.to(self.device)

            if is_train:
                with autocast(enabled=self.config.AMP):
                    y_pred = self.model(X)
                    loss = self.criterion(F.log_softmax(y_pred, dim=1), y)
                if self.config.GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                with torch.no_grad():
                    y_pred = self.model(X)
                    loss = self.criterion(F.log_softmax(y_pred, dim=1), y)
                if self.config.GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                
                predicts_record.append(y_pred.to('cpu').numpy())
            
            loss_meter.update(loss.item(), y.size(0))
            end = time()

            if (step % self.config.PRINT_FREQ == 0) or (step == (len_loader - 1)):
                lr = self.scheduler.get_last_lr()[0]
                info = f"Epoch {epoch_id + 1} [{step}/{len_loader}] | {mode} Loss: {loss_meter.avg:.4f}"
                if is_train:
                    info += f" Grad: {grad_norm:.4f} LR: {lr:.4e}"
                info += f" | Elapse: {end - start:.2f}s"
                print(info)

        if not is_train:
            predicts_record = np.concatenate(predicts_record)
            
        return loss_meter.avg, predicts_record


def evaluate_oof(oof_df):
    '''
    Evaluate the out-of-fold dataframe using KL Divergence (torch and kaggle)
    '''
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[TARGETS].values.astype('float32'))
    preds = F.log_softmax(
        torch.tensor(oof_df[TARGETS_PRED].values.astype('float32'), requires_grad=False),
        dim=1
    )
    kl_torch = kl_loss(preds, labels).item()

    return kl_torch


class HMSPredictor:

    def __init__(self, output_dir, model_config, k_fold=5):

        self.output_dir = output_dir
        self.model_config = model_config
        self.k_fold = k_fold

        self.model_name = self.model_config.MODEL_NAME

        self.logger = get_logger(self.output_dir, f"{self.model_name}_train.log")
        self.__log_init_info()

        seed_everything(self.model_config.SEED)

    def get_model(self, pretrained=True):
        backbone = self.model_config.MODEL_BACKBONE

        if "efficientnet" in backbone:
            return CustomEfficientNET(self.model_config, num_classes=6, pretrained=pretrained)
        elif "vit" in backbone:
            return CustomVITMAE(self.model_config, num_classes=6, pretrained=pretrained)
        elif "dual" in backbone:
            return DualEncoderModel(self.model_config, num_classes=6, pretrained=pretrained)
        else:
            return None

    def __log_init_info(self):
        self.logger.info(f"{'*' * 100}")
        self.logger.info(f"Script Start: {ctime()}")
        self.logger.info(f"Model Configurations:")
        for key, value in self.model_config.__dict__.items():
            if not key.startswith("__"):
                self.logger.info(f"{key}: {value}")
        self.logger.info(f"{'*' * 100}")

    def _plot_loss(self, loss_per_fold, stage):

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid(True)
        line_colors = plt.get_cmap('tab10').colors
        for i, loss in enumerate(loss_per_fold):
            ax.plot(loss['train'], 'o-', c=line_colors[i], label=f"Train {i}")
            ax.plot(loss['valid'], 'o:', c=line_colors[i], label=f"Valid {i}")
        ax.set_ylim(0, 1)
        ax.set_title(self.model_config.MODEL_NAME + f" Loss Plot ({stage})")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{self.model_config.MODEL_NAME}_loss_plot_{stage}.png"))
        plt.close(fig)

    def prepare_k_fold(self, df):

        kf = KFold(n_splits=self.k_fold)
        unique_spec_id = df['spectrogram_id'].unique()
        df['fold'] = self.k_fold 

        for fold, (train_index, valid_index) in enumerate(kf.split(unique_spec_id)):
            df.loc[df['spectrogram_id'].isin(unique_spec_id[valid_index]), 'fold'] = fold

        return df

    def log_validation_info(self, fold, valid_folds, tik):
        kl_torch_easy = evaluate_oof(valid_folds[valid_folds['easy_or_hard'] == "easy"])
        kl_torch_hard = evaluate_oof(valid_folds[valid_folds['easy_or_hard'] == "hard"])
        info = f"{'=' * 100}\nFold {fold} Valid Loss: (Easy) {kl_torch_easy:.4f} | (Hard) {kl_torch_hard:.4f}\n"
        info += f"Elapse: {(time() - tik) / 60:.2f} min \n{'=' * 100}"
        self.logger.info(info)
 
    def train_model(self, train_easy, train_hard, all_specs, all_eegs):
        
        self.logger.info(f"Train Easy: {train_easy.shape} | Train Hard: {train_hard.shape}")

        train_easy = self.prepare_k_fold(train_easy)
        train_hard = self.prepare_k_fold(train_hard)

        oof_stage_1, oof_stage_2 = pd.DataFrame(), pd.DataFrame()
        loss_history_1, loss_history_2 = [], []

        tik_total = time()
        for fold in range(self.k_fold):
            tik = time()

            train_folds_easy = train_easy[train_easy['fold'] != fold].reset_index(drop=True)
            train_folds_easy['train_label'] = 1
            valid_folds_easy = train_easy[train_easy['fold'] == fold].reset_index(drop=True)
            valid_folds_easy['easy_or_hard'] = "easy"

            train_folds_hard = train_hard[train_hard['fold'] != fold].reset_index(drop=True)
            train_folds_hard['train_label'] = 1
            valid_folds_hard = train_hard[train_hard['fold'] == fold].reset_index(drop=True)
            valid_folds_hard['easy_or_hard'] = "hard"
            
            valid_folds = pd.concat([valid_folds_easy, valid_folds_hard], axis=0).reset_index(drop=True)

            valid_folds = valid_folds.merge(train_folds_easy[['eeg_id', 'train_label']], on='eeg_id', how='left')
            valid_folds = valid_folds[valid_folds['train_label'] != 1].reset_index(drop=True)
            valid_folds.drop('train_label', axis=1, inplace=True)

            valid_folds = valid_folds.merge(train_folds_hard[['eeg_id', 'train_label']], on='eeg_id', how='left')
            valid_folds = valid_folds[valid_folds['train_label'] != 1].reset_index(drop=True)
            valid_folds.drop('train_label', axis=1, inplace=True)
            
            # STAGE 1
            self.logger.info(f"{'=' * 100}\nFold: {fold} || Valid size {valid_folds.shape[0]} \n{'=' * 100}")
            self.logger.info(f"- First Stage ")
            valid_predicts, loss_records = self._train_fold(
                fold, train_folds_easy, valid_folds, all_specs, all_eegs, stage=1, checkpoint=None)

            loss_history_1.append(loss_records)
            valid_folds[TARGETS_PRED] = valid_predicts
            oof_stage_1 = pd.concat([oof_stage_1, valid_folds], axis=0).reset_index(drop=True)
            self.log_validation_info(fold, valid_folds, tik)
            self._plot_loss(loss_history_1, stage="1")

            # STAGE 2
            tik = time()
            self.logger.info(f"- Second Stage ")
            check_point = os.path.join(
                self.output_dir,
                f"{self.model_name}_fold_{fold}_stage_1.pth"
            )
            self.logger.info(f"Use Checkpoint: {check_point.split('/')[-1]}")

            valid_predicts, loss_records = self._train_fold(
                fold, train_folds_hard, valid_folds, all_specs, all_eegs, stage=2, checkpoint=check_point)

            loss_history_2.append(loss_records)
            valid_folds[TARGETS_PRED] = valid_predicts
            oof_stage_2 = pd.concat([oof_stage_2, valid_folds], axis=0).reset_index(drop=True)
            self.log_validation_info(fold, valid_folds, tik)
            self._plot_loss(loss_history_2, stage="2")

            oof_stage_1.to_csv(os.path.join(self.output_dir, f"{self.model_name}_oof_1.csv"), index=False)
            oof_stage_2.to_csv(os.path.join(self.output_dir, f"{self.model_name}_oof_2.csv"), index=False)

        info = f"{'=' * 100}\nTraining Complete!\n"
        for i, oof_df in enumerate([oof_stage_1, oof_stage_2]):
            cv_results = evaluate_oof(oof_df)
            info += f"CV Result (Stage={i + 1}): {cv_results}\n"
        info += f"Elapse: {(time() - tik_total) / 60:.2f} min \n{'=' * 100}"
        self.logger.info(info)


    def _train_fold(self, fold_id, train_folds, valid_folds, all_specs, all_eegs, stage=1, checkpoint=None):

        model = self.get_model()

        train_dataset = CustomDataset(
            train_folds, TARGETS, self.model_config, all_specs, all_eegs, mode="train")

        valid_dataset = CustomDataset(
            valid_folds, TARGETS, self.model_config, all_specs, all_eegs, mode="valid")

        # ======== DATALOADERS ==========
        loader_kwargs = {
            "batch_size": self.model_config.BATCH_SIZE,
            "num_workers": self.model_config.NUM_WORKERS,
            "pin_memory": True,
            "shuffle": False,
        }
        train_loader = DataLoader(train_dataset, drop_last=True, **loader_kwargs)
        valid_loader = DataLoader(valid_dataset, drop_last=False, **loader_kwargs)

        trainer = Trainer(model, self.model_config, self.logger)
        best_weights, best_preds, loss_records = trainer.train(train_loader, valid_loader, from_checkpoint=checkpoint)

        save_model_name = f"{self.model_name}_fold_{fold_id}_stage_{stage}.pth"
        torch.save(best_weights, os.path.join(self.output_dir, save_model_name))

        del train_dataset, valid_dataset, train_loader, valid_loader
        torch.cuda.empty_cache()
        gc.collect()

        return best_preds, loss_records