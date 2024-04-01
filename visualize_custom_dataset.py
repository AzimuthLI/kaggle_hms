
import os 
import numpy as np 
import matplotlib.pyplot as plt

from my_hms_models import * 
from train_hms_models import get_non_overlap
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# define the paths
paths = KagglePaths if os.path.exists(KagglePaths.OUTPUT_DIR) else LocalPaths


if __name__ == "__main__":
    
    def plot_func(x, y, save_name):
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flatten()):
            im = ax.imshow(x[i, :, :], cmap="viridis")
            plt.colorbar(im, ax=ax, orientation='horizontal')

        fig.suptitle(f"Target: {y}")
        fig.tight_layout()
        fig.savefig(f"{paths.OUTPUT_DIR}/{save_name}.png")

    
    # Load the data
    train_csv = pd.read_csv(paths.TRAIN_CSV)
    TARGETS = train_csv.columns[-6:]

    # Get the non-overlapping data
    df_train = get_non_overlap(train_csv, TARGETS, ModelConfig.GROUP_KEYS, calc_method='simple')

    print(f"{'-'*100}\nTrain Data\n{'-'*100}")
    print(df_train.head(10))
    print(f"{'-'*100}\nTrain Data Shape: {df_train.shape}\n{'-'*100}")

    all_specs = np.load(paths.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
    all_eegs = np.load(paths.PRE_LOADED_EEGS, allow_pickle=True).item()
    
    augment_dataset = CustomDataset(
        df_train, TARGETS, ModelConfig, all_specs, all_eegs, augment=True, mode='train')
    
    original_dataset = CustomDataset(
        df_train, TARGETS, ModelConfig, all_specs, all_eegs, augment=False, mode='train')
    
    visual_ids = np.random.randint(0, len(df_train), 5)
    for visual_id in visual_ids:
        print(visual_id)
        aug = ['with_aug', 'without_aug']
        for i, ds in enumerate([augment_dataset, original_dataset]):
            X, y = ds[visual_id]
            print(f"{aug[i]}: X: {X.shape}, y: {y.tolist()}")
            plot_func(X, y, f"raw_spectrogram_{visual_id}_{aug[i]}")

    dataloader = DataLoader(augment_dataset, batch_size=16, shuffle=True, num_workers=4)
    for X, y in dataloader:
        print(f"X: {X.shape}, y: {y.shape}")
        break
    
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"{'-'*10}\nDevice: {DEVICE}\n{'-'*10}")

    # dataloader = DataLoader(augment_dataset, batch_size=16, shuffle=True, num_workers=4)
    # model = CustomModel(ModelConfig)
    # model.to(DEVICE)

    # for X, y in dataloader:
    #     X = X.to(DEVICE)
    #     y = y.to(DEVICE)
    #     print(f"X: {X.shape}, y: {y.shape}")
        
    #     y_pred = model(X)


    
    