# Regular imports
import os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import pywt, librosa

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from config_hms_models import ModelConfig, KagglePaths, LocalPaths
from my_hms_models import CustomModel, CustomDataset, transform_spectrogram

# Constants
USE_WAVELET = None 

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

TARGET_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# FUNCTIONS
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


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret


def spectrogram_from_eeg(parquet_path, display=False, offset=None):
    
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)

    if offset is None:
        middle = (len(eeg)-10_000)//2
        eeg = eeg.iloc[middle:middle+10_000]
    else:
        eeg = eeg.iloc[offset:offset+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4), dtype='float32')
    
    signals = []

    for k in range(4):

        COLS = FEATS[k]
        for kk in range(4):
            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values
            # FILL NANS
            x = np.nan_to_num(x, nan=np.nanmean(x)) if np.isnan(x).mean() < 1 else x*0
            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(
                y=x, 
                sr=200, 
                hop_length=len(x)//256,
                n_fft=1024, 
                n_mels=128, 
                fmin=0, 
                fmax=20, 
                win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
        
    if display:

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        for i, ax in enumerate(axes.flatten()):
            im = ax.imshow(img[:, :, i], origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(NAMES[i])
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000), signals[k]+offset, label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.show()
        
    return img


def inference_function(test_loader, model):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy()) 
                
    prediction_dict["predictions"] = np.concatenate(preds) 
    return prediction_dict



if __name__ == "__main__":

    # define the paths
    paths = KagglePaths if os.path.exists(KagglePaths.OUTPUT_DIR) else LocalPaths

    model_weights = [x for x in glob(paths.OUTPUT_DIR + "*.pth")]
    print(f"{'-'*10}\nModel Weights")
    for mw in model_weights:
        print(mw)
    print(f"{'-'*10}")

    test_df = pd.read_csv(paths.TEST_CSV)
    print('Test shape',test_df.shape)
    print(test_df.head())
    
    # READ ALL SPECTROGRAMS
    print(f"{'-'*10}\nReading All Spectrograms\n{'-'*10}")
    paths_spectrograms = glob(os.path.join(paths.TEST_SPECTROGRAMS, "*.parquet"))
    print(f'There are {len(paths_spectrograms)} spectrogram parquets')
    all_spectrograms = {}

    for file_path in tqdm(paths_spectrograms):
        aux = pd.read_parquet(file_path)
        name = int(file_path.split("/")[-1].split('.')[0])
        all_spectrograms[name] = aux.iloc[:,1:].values
        del aux
        
    if ModelConfig.VISUALIZE:
        idx = np.random.randint(0, len(paths_spectrograms))
        spectrogram_path = paths_spectrograms[idx]
        plot_spectrogram(spectrogram_path)

    # READ ALL EEG SPECTROGRAMS
    print(f"{'-'*10}\nReading All EEG Spectrograms\n{'-'*10}")
    paths_eegs = glob(os.path.join(paths.TEST_EEGS, "*.parquet"))
    print(f'There are {len(paths_eegs)} EEG spectrograms')
    all_eegs = {}
    counter = 0

    for file_path in tqdm(paths_eegs):
        eeg_id = file_path.split("/")[-1].split(".")[0]
        eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
        all_eegs[int(eeg_id)] = eeg_spectrogram
        counter += 1


    # INFERENCE
    print(f"{'-'*10}\nInference Starts...\n{'-'*10}")
    predictions = []
    for model_weight in model_weights:

        test_dataset = CustomDataset(test_df, TARGET_COLS, ModelConfig, all_spectrograms, all_eegs, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=ModelConfig.NUM_WORKERS,
            pin_memory=True, 
            drop_last=False
        )

        model = CustomModel(ModelConfig, pretrained=False)
        checkpoint = torch.load(model_weight)
        model.load_state_dict(checkpoint)

        model.to(DEVICE)
        prediction_dict = inference_function(test_loader, model)
        predictions.append(prediction_dict["predictions"])
        torch.cuda.empty_cache()
        gc.collect()
        
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)

    sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
    sub[TARGET_COLS] = predictions

    # Sanity check
    sum_to_one = sub[TARGET_COLS].sum(axis=1).values[0]

    print(sum_to_one)

    if sum_to_one == 1.0:
        print("All predictions sum to 1.0. Passed the sanity check.")
        print(f'Submissionn shape: {sub.shape}')
        print(sub.head())
        sub.to_csv('submission.csv', index=False)

    else:
        raise ValueError("Predictions do not sum to 1.0. Please check the predictions.")