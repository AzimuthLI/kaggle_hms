import os, sys, gc 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import pytorch_lightning as pl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold

from kl_div import score as kl_score 


USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = False

TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS2 = {x: y for y, x in TARS.items()}


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
    train.columns = ['spec_id', 'spec_offset_min', 'spec_offset_max', 'patient_id'] + tgt_list + ['target']
    train = train.reset_index(drop=False)

    train[tgt_list] = train[tgt_list].div(train[tgt_list].sum(axis=1), axis='index')

    return train


class DataGenerator(Dataset):
    
    def __init__(self, data, specs, targets, mode='train'): 

        self.data = data
        self.mode = mode
        self.specs = specs
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.__getitems__([index])
    
    def __getitems__(self, indices):
        X, y = self._generate_data(indices)
     
        if self.mode == 'train':
            return list(zip(X, y))
        else:
            return X
    
    def _generate_data(self, indexes):
        # This dataloader outputs 4 spectrogram images as a 4 channel image of size 128x256x4 per train sample
        # 4 channels: LL, LR, RL, RR
        # Note: only kaggle spectrograms are used --> only 4 channels 
        X = np.zeros( (len(indexes), 128, 256, 4), dtype='float32')
        y = np.zeros( (len(indexes), 6), dtype='float32')
        img = np.ones( (128, 256), dtype='float32')
        
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == 'test': 
                r = 0
            else: 
                r = int((row['spec_offset_min'] + row['spec_offset_max'])//4)

            for k in range(4):
                # EXTRACT 300 ROWS OF SPECTROGRAM
                # --> r: rows; k: frequency bands (100 bands per channel)
                img = self.specs[row.spec_id][r:r+300, k*100:(k+1)*100].T
                
                # LOG TRANSFORM SPECTROGRAM
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                
                # STANDARDIZE PER IMAGE
                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img - m) / (s + ep)
                img = np.nan_to_num(img, nan=0.0)
                
                # CROP TO 256 TIME STEPS
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0
        
            if self.mode != 'test':
                y[j,] = row[self.targets].values
            
        return X, y


def plot_dataloader(dataloader, df_train, rows=2, cols=3, batches=2, show_plot=False):

    for i, (x, y) in enumerate(dataloader):
        fig, axes = plt.subplots(rows, cols, figsize=(20, 8), sharex=True, sharey=True)
        for j in range(rows):
            for k in range(cols):
                ax = axes.flatten()[j*cols + k]
                t = y[j*cols + k]
                img = torch.flip(x[j*cols+k, :, :, 0], (0,))
                mn = img.flatten().min()
                mx = img.flatten().max()
                img = (img-mn)/(mx-mn)
                ax.imshow(img)

                tars = f'[{t[0]:0.2f}]'
                for s in t[1:]:
                    tars += f', {s:0.2f}'
                eeg = df_train.eeg_id.values[i*32+j*cols+k]
                ax.set_title(f'EEG = {eeg}\nTarget = {tars}',size=12)

                # plt.yticks([])
                if j == rows-1:
                    ax.set_xlabel('Frequencies (Hz)',size=12)
                
                if k == 0:
                    ax.set_ylabel('Time (sec)',size=12)

        if show_plot:
            plt.show(block=True)
        else:
            fig.savefig(f'dataloader_batch_{i}.png')

        if i == batches-1:
            break


class EEGEffnetB0(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights)
        # self.base_model.load_state_dict(torch.load(WEIGHTS_FILE))
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, 6, dtype=torch.float32)
        self.prob_out = nn.Softmax()
        
    def forward(self, x):
        x1 = [x[:, :, :, i:i+1] for i in range(4)]
        x1 = torch.concat(x1, dim=1)
        # x2 = [x[:, :, :, i+4:i+5] for i in range(4)]
        # x2 = torch.concat(x2, dim=1)
        
        # if USE_KAGGLE_SPECTROGRAMS & USE_EEG_SPECTROGRAMS:
        #     x = torch.concat([x1, x2], dim=2)
        # elif USE_EEG_SPECTROGRAMS:
        #     x = x2
        # else:
        #     x = x1
        x = torch.concat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        
        out = self.base_model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = F.log_softmax(out, dim=1)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        loss = kl_loss(out, y)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return F.softmax(self(batch), dim=1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train_and_valid(df_train, targets, n_splits=5, batch_size=32, num_workers=3, max_epochs=4, version=0):

    all_oof = []
    all_true = []
    valid_loaders = []

    gkf = GroupKFold(n_splits=n_splits)

    for i, (train_index, valid_index) in enumerate(gkf.split(df_train, df_train.target, df_train.patient_id)):  
        print('#'*25)
        print(f'### Fold {i+1}')
        
        train_ds = DataGenerator(df_train.iloc[train_index], all_specs, targets=targets, mode='train')
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        
        valid_ds = DataGenerator(df_train.iloc[valid_index], all_specs, targets=targets, mode='valid')
        valid_loader = DataLoader(valid_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        
        print(f'### Train size: {len(train_index)}, Valid size: {len(valid_index)}')
        print('#'*25)
        
        trainer = pl.Trainer(max_epochs=max_epochs)
        model = EEGEffnetB0()
    
        trainer.fit(model=model, train_dataloaders=train_loader)
        trainer.save_checkpoint(f'EffNet_v{version}_f{i}.ckpt')

        valid_loaders.append(valid_loader)
        all_true.append(df_train.iloc[valid_index][targets].values)

        del trainer, model
        gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(n_splits):
        print('#'*25)
        print(f'### Validating Fold {i+1}')

        ckpt_file = f'EffNet_v{version}_f{i}.ckpt'
        model = EEGEffnetB0.load_from_checkpoint(ckpt_file)
        model.to(device).eval()
        with torch.inference_mode():
            for val_batch in valid_loaders[i]:
                val_batch = val_batch.to(device)
                oof = torch.softmax(model(val_batch), dim=1).cpu().numpy()
                all_oof.append(oof)

        del model
        gc.collect()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    df_oof = pd.DataFrame(all_oof.copy())
    df_oof['id'] = np.arange(len(oof))

    df_true = pd.DataFrame(all_true.copy())
    df_true['id'] = np.arange(len(df_true))

    cv = kl_score(solution=df_true, submission=oof, row_id_column_name='id')

    print('CV Score KL-Div for EffNet_v{version} =', cv)


    return df_oof, df_true


if __name__ == "__main__":

    train_csv = pd.read_csv('../train.csv')
    TARGETS = train_csv.columns[-6:]

    df_train = get_non_overlap(train_csv, TARGETS)

    print('Train shape: ', df_train.shape )
    print('Targets: ', list(TARGETS))

    print(df_train.head())

    df_specs = pd.read_parquet('../all_specs.parquet')
    all_specs = {
        int(k): v.drop(['spec_id', 'time'], axis=1).values for k, v in df_specs.groupby('spec_id')
        }
    
    dataset = DataGenerator(df_train, all_specs, targets=TARGETS ,mode='train')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    plot_dataloader(dataloader, df_train, rows=2, cols=3, batches=2, show_plot=False)

    del dataset, dataloader
    gc.collect()

    train_params = {
        'n_splits': 5,
        'batch_size': 32,
        'num_workers': 3,
        'max_epochs': 4,
        'version': 0
    }

    df_oof, df_true = train_and_valid(df_train, TARGETS, **train_params)

    