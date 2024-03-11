from tqdm.notebook import tqdm
import logging
import random
import numpy as np
import pandas as pd
import os, gc
from time import time, ctime

from scipy.special import rel_entr, softmax
from sklearn.model_selection import GroupKFold, KFold
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from kl_divergence import score as kaggle_score
from engine_hms_model import CustomDataset, CustomModel

# CONSTANTS
BRAIN_ACTIVITY = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
TARGETS = [f"{lb}_vote" for lb in BRAIN_ACTIVITY]
TARGETS_PRED = [f"{lb}_pred" for lb in BRAIN_ACTIVITY]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def cal_entropy(row, tgt_list):
    nc = len(tgt_list)
    uniform_list = [1 / nc for i in range(nc)]
    return sum(rel_entr(uniform_list, row[tgt_list].astype('float32').values + 1e-5))


def gen_non_overlap_samples(df_csv, targets):
    # Reference Discussion:
    # https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467021

    # train and validate using only 1 crop per eeg_id
    # Simple method: simple average of the votes as probability
    # Weighted method: entropy weighted average of the votes as probability

    tgt_list = targets.tolist()
    brain_activity = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    n_classes = len(brain_activity)

    agg_dict = {
        'spectrogram_id': 'first',
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'patient_id': 'first',
        'expert_consensus': 'first'
    }

    groupby = df_csv.groupby(['eeg_id'] + tgt_list)
    train = groupby.agg(agg_dict)
    train = train.reset_index()
    train.columns = ['_'.join(col).strip() for col in train.columns.values]
    train.columns = ["eeg_id"] + tgt_list + ['spectrogram_id', 'min', 'max', 'patient_id', 'target']
    
    vote_sum = train[tgt_list]
    train[tgt_list] = vote_sum.div(vote_sum.sum(axis=1), axis=0)
    # from scipy.special import softmax
    # train[tgt_list] = softmax(train[tgt_list].values, axis=1)

    return train


def load_kaggle_data(paths, split_entropy=5.5):
    train_csv = pd.read_csv(paths.TRAIN_CSV)
    targets = train_csv.columns[-6:]

    train_csv['entropy'] = train_csv.apply(cal_entropy, axis=1, tgt_list=targets)
    train_csv['total_votes'] = train_csv[targets].sum(axis=1)

    easy_csv = train_csv[train_csv['entropy'] >= split_entropy].copy().reset_index(drop=True)
    hard_csv = train_csv[train_csv['entropy'] < split_entropy].copy().reset_index(drop=True)

    train_easy = gen_non_overlap_samples(easy_csv, targets)
    train_hard = gen_non_overlap_samples(hard_csv, targets)

    all_specs = np.load(paths.PRE_LOADED_SPECTROGRAMS, allow_pickle=True).item()
    all_eegs = np.load(paths.PRE_LOADED_EEGS, allow_pickle=True).item()

    return train_easy, train_hard, all_specs, all_eegs


class Trainer:

    def __init__(self,
                 model,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader,
                 device,
                 config,
                 logger,
                 check_point_path=None):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.optimizer, self.scheduler = self.get_optimizer(model, config)
        self.check_point_path = check_point_path
        self.logger = logger

        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def get_optimizer(self, model, config):

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config.WEIGHT_DECAY)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-4,
            epochs=config.EPOCHS,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=100,
        )
        # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=3)

        return optimizer, scheduler

    def compute_loss(self, y_pred, y_true):
        # criterion = nn.KLDivLoss(reduction="batchmean")
        # criterion = nn.CrossEntropyLoss()
        return self.criterion(F.log_softmax(y_pred, dim=1), y_true)

    def train(self):

        if self.check_point_path is not None:
            self.load_checkpoint(self.check_point_path)

        self.model.to(self.device)
        best_loss = np.inf
        best_model_weights = None
        loss_records = {"train": [], "valid": []}

        for epoch in range(self.config.EPOCHS):
            start_time = time()

            self.model.train()
            train_loss = self._train_epoch(epoch)
            valid_loss, valid_preds = self._valid_epoch(epoch)

            elapsed = time() - start_time
            info = f"{'-' * 100}\nEpoch {epoch + 1} - "
            info += f"Average Train Loss: {train_loss:.4f} | Average Valid Loss: {valid_loss:.4f} | Time: {elapsed:.2f}s"
            self.logger.info(info)

            loss_records["train"].append(train_loss)
            loss_records["valid"].append(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model_weights = self.model.state_dict()
                best_valid_preds = valid_preds
                self.logger.info(f"Best model found in epoch {epoch + 1} | valid loss: {best_loss:.4f}")

        return best_model_weights, best_valid_preds, loss_records

    def _train_epoch(self, epoch):

        self.model.train()

        len_loader = len(self.train_loader)
        scaler = GradScaler(enabled=self.config.AMP)
        losses = AverageMeter()
        start = end = time()

        with tqdm(self.train_loader, unit="batch", desc='Train') as pbar:

            for step, (X, y) in enumerate(pbar):

                X, y = X.to(self.device), y.to(self.device)
                batch_size = y.size(0)

                with autocast(enabled=self.config.AMP):
                    y_pred = self.model(X)
                    loss = self.compute_loss(y_pred, y)

                if self.config.GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS

                losses.update(loss.item(), batch_size)
                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)

                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                end = time()

                if step % self.config.PRINT_FREQ == 0 or step == (len_loader - 1):
                    lr = self.scheduler.get_last_lr()[0]
                    info = f"Epoch: [{epoch + 1}][{step}/{len_loader}]"
                    info += f"Elapsed {(end - start):.2f}s | Loss: {losses.avg:.4f} Grad: {grad_norm:.4f} LR: {lr:.4e}"
                    print(info)

        return losses.avg

    def _valid_epoch(self, epoch):

        self.model.eval()

        len_loader = len(self.valid_loader)
        losses = AverageMeter()
        start = end = time()

        predicts = []

        with tqdm(self.valid_loader, unit="batch", desc='Valid') as pbar:

            for step, (X, y) in enumerate(pbar):

                X, y = X.to(self.device), y.to(self.device)
                batch_size = y.size(0)

                with torch.no_grad():
                    y_pred = self.model(X)
                    loss = self.compute_loss(y_pred, y)

                if self.config.GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS

                losses.update(loss.item(), batch_size)
                predicts.append(y_pred.to('cpu').numpy())
                end = time()

                if step % self.config.PRINT_FREQ == 0:
                    info = f"Epoch: [{epoch + 1}][{step}/{len_loader}]"
                    info += f"Elapsed {(end - start):.2f}s | Loss: {losses.avg:.4f}"
                    print(info)

        torch.cuda.empty_cache()
        gc.collect()

        return losses.avg, np.concatenate(predicts)


def evaluate_oof(oof_df):
    '''
    Evaluate the out-of-fold dataframe using KL Divergence (torch and kaggle)
    '''
    softmax = nn.Softmax(dim=1)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[TARGETS].values)
    preds = torch.tensor(oof_df[TARGETS_PRED].values)
    preds = F.log_softmax(preds, dim=1)
    kl_torch = kl_loss(preds, labels)

    # solution = oof_df[['eeg_id'] + TARGETS].copy()
    # submission = oof_df[['eeg_id'] + TARGETS_PRED].copy()
    # submission = submission.rename(columns={f"{lb}_pred": f"{lb}_vote" for lb in BRAIN_ACTIVITY})
    # submission[TARGETS] = softmax(
    #     torch.tensor(submission[TARGETS].values.astype('float32'))
    #     ).numpy()
    # kl_kaggle = kaggle_score(solution, submission, 'eeg_id')

    return kl_torch  # , kl_kaggle


class HMSPredictor:

    def __init__(self, job_config, model_config):

        self.job_config = job_config
        self.model_config = model_config
        self.k_fold = job_config.K_FOLD

        self.logger = get_logger(self.job_config.OUTPUT_DIR, f"{self.model_config.MODEL_NAME}_loss.log")
        self.__log_init_info()

    def get_model(self, pretrained=True):
        return CustomModel(self.model_config, num_classes=6, pretrained=pretrained)

    def __log_init_info(self):

        self.logger.info(f"{'*' * 100}")
        self.logger.info(f"Script Start: {ctime()}")
        self.logger.info(f"Initializing HMS Predictor...")
        self.logger.info(f"Model Name: {self.model_config.MODEL_NAME}")
        self.logger.info(f"Drop Rate: {self.model_config.DROP_RATE}")
        self.logger.info(f"Drop Path Rate: {self.model_config.DROP_PATH_RATE}")
        self.logger.info(f"Augment: {self.model_config.AUGMENT}")
        if self.model_config.AUGMENT:
            self.logger.info(f"Augmentations: {self.model_config.AUGMENTATIONS}")
        self.logger.info(f"Enropy Split: {self.job_config.ENTROPY_SPLIT}")
        self.logger.info(f"Device: {DEVICE}")
        self.logger.info(f"Output Dir: {self.job_config.OUTPUT_DIR}")
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

        ax.set_title(self.model_config.MODEL_NAME + f" Loss Plot ({stage})")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(self.job_config.OUTPUT_DIR, f"{self.model_config.MODEL_NAME}_loss_plot_{stage}.png"))
        plt.close(fig)

    def load_train_data(self):

        train_easy, train_hard, all_specs, all_eegs = load_kaggle_data(
            self.job_config.PATHS,
            self.job_config.ENTROPY_SPLIT
        )

        return train_easy, train_hard, all_specs, all_eegs

    def train_folds(self, train_easy, train_hard, all_specs, all_eegs):

        # split easy and hard data into k-folds
        unique_spec_id_easy = train_easy["spectrogram_id"].unique()
        train_easy["fold"] = self.k_fold
        kf = KFold(n_splits=self.k_fold)
        for fold, (train_spec_index, valid_spec_index) in enumerate(kf.split(unique_spec_id_easy)):
            valid_spec_id = unique_spec_id_easy[valid_spec_index]
            valid_index = train_easy[train_easy.spectrogram_id.isin(valid_spec_id)].index
            train_easy.loc[valid_index, "fold"] = int(fold)

        unique_spec_id_hard = train_hard["spectrogram_id"].unique()
        train_hard["fold"] = self.k_fold
        kf = KFold(n_splits=self.k_fold)
        for fold, (train_spec_index, valid_spec_index) in enumerate(kf.split(unique_spec_id_hard)):
            valid_spec_id = unique_spec_id_hard[valid_spec_index]
            valid_index = train_hard[train_hard.spectrogram_id.isin(valid_spec_id)].index
            train_hard.loc[valid_index, "fold"] = int(fold)

        def get_valid_fold(fold):
            valid_folds_easy = train_easy[train_easy['fold'] == fold].copy().reset_index(drop=True)
            valid_folds_easy['easy_or_hard'] = "easy"

            valid_folds_hard = train_hard[train_hard['fold'] == fold].copy().reset_index(drop=True)
            valid_folds_hard['easy_or_hard'] = "hard"

            return pd.concat([valid_folds_easy, valid_folds_hard], axis=0).reset_index(drop=True)

        valid_folds_list = [get_valid_fold(fold) for fold in range(self.k_fold)]

        oof_df1, oof_df2 = pd.DataFrame(), pd.DataFrame()
        loss_per_fold_1, loss_per_fold_2 = [], []

        tik_total = time()
        for fold in range(self.k_fold):
            tik = time()

            train_folds_easy = train_easy[train_easy['fold'] != fold].copy().reset_index(drop=True)
            train_folds_hard = train_hard[train_hard['fold'] != fold].copy().reset_index(drop=True)

            valid_folds = valid_folds_list[fold]

            # first stage with easy samples
            self.logger.info(f"{'=' * 100}\nFold: {fold} || Valid size {valid_folds.shape[0]} \n{'=' * 100}")
            self.logger.info(f"- First Stage ")
            valid_predicts, loss_records = self._train_fold(
                fold, train_folds_easy, valid_folds, all_specs, all_eegs, stage=1, check_point=None)

            loss_per_fold_1.append(loss_records)

            valid_folds[TARGETS_PRED] = valid_predicts
            kl_torch_easy = evaluate_oof(valid_folds[valid_folds['easy_or_hard'] == "easy"])
            kl_torch_hard = evaluate_oof(valid_folds[valid_folds['easy_or_hard'] == "hard"])
            info = f"{'=' * 100}\nFold {fold} Valid Loss: (Easy) {kl_torch_easy:.4f} | (Hard) {kl_torch_hard:.4f}\n"
            info += f"Elapse: {(time() - tik) / 60:.2f} min \n{'=' * 100}"
            self.logger.info(info)

            oof_df1 = pd.concat([oof_df1, valid_folds], axis=0).reset_index(drop=True)
            self._plot_loss(loss_per_fold_1, stage="1")

            # second stage with hard samples
            tik = time()
            self.logger.info(f"- Second Stage ")
            check_point = os.path.join(
                self.job_config.OUTPUT_DIR,
                f"{self.model_config.MODEL_NAME}_fold_{fold}_stage_1.pth"
            )
            self.logger.info(f"Use Checkpoint: {check_point.split('/')[-1]}")

            valid_predicts, loss_records = self._train_fold(
                fold, train_folds_hard, valid_folds, all_specs, all_eegs, stage=2, check_point=check_point)

            loss_per_fold_2.append(loss_records)

            valid_folds[TARGETS_PRED] = valid_predicts
            kl_torch_easy = evaluate_oof(valid_folds[valid_folds['easy_or_hard'] == "easy"])
            kl_torch_hard = evaluate_oof(valid_folds[valid_folds['easy_or_hard'] == "hard"])
            info = f"{'=' * 100}\nFold {fold} Valid Loss: (Easy) {kl_torch_easy:.4f} | (Hard) {kl_torch_hard:.4f}\n"
            info += f"Elapse: {(time() - tik) / 60:.2f} min \n{'=' * 100}"
            self.logger.info(info)

            oof_df2 = pd.concat([oof_df2, valid_folds], axis=0).reset_index(drop=True)
            self._plot_loss(loss_per_fold_2, stage="2")

        oof_df1.to_csv(os.path.join(self.job_config.OUTPUT_DIR, f"{self.model_config.MODEL_NAME}_oof_1.csv"),
                       index=False)
        oof_df2.to_csv(os.path.join(self.job_config.OUTPUT_DIR, f"{self.model_config.MODEL_NAME}_oof_2.csv"),
                       index=False)

        info = f"{'=' * 100}\nTraining Complete!\n"
        for i, oof_df in enumerate([oof_df1, oof_df2]):
            cv_results = evaluate_oof(oof_df)
            info += f"CV Result (Stage={i + 1}): {cv_results}\n"

        info += f"Elapse: {(time() - tik_total) / 60:.2f} min \n{'=' * 100}"
        self.logger.info(info)

    def _train_fold(self, fold_id, train_folds, valid_folds, all_specs, all_eegs, stage=1, check_point=None):

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

        trainer = Trainer(model, train_loader, valid_loader, None, DEVICE, self.model_config, self.logger, check_point)

        best_model_weights, valid_preds, loss_records = trainer.train()

        save_model_name = f"{self.model_config.MODEL_NAME}_fold_{fold_id}_stage_{stage}.pth"
        torch.save(best_model_weights, os.path.join(self.job_config.OUTPUT_DIR, save_model_name))

        del train_dataset, valid_dataset, train_loader, valid_loader
        torch.cuda.empty_cache()
        gc.collect()

        return valid_preds, loss_records
