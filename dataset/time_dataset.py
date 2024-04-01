import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import itertools

class TsDataset(Dataset):
    """Time series dataset."""
    def __init__(self, csv_file, normalize=True):
        """
        Args:
            csv_file (string): path to csv file
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = pd.read_csv(csv_file)
        df = df.set_index('Label').fillna(0)

        groups = df.groupby(df.index, sort=False)
        group_normal = groups.get_group('Normal')
        groups_abnormal = itertools.islice(df.groupby(df.index, sort=False), 1, None)

        data_normal = torch.from_numpy(group_normal.values).permute(1, 0).float()
        data_abnormal = torch.from_numpy(np.array([group[1] for group in groups_abnormal])).permute(2, 0, 1).float()

        len1, len2 = data_normal.size(1), data_abnormal.size(2)

        min_len = min(len1, len2)

        data_normal = data_normal[:, :min_len]  # Truncate the longer sequence
        data_abnormal = data_abnormal[:, :, :min_len]

        self.data_abnormal = self.normalize(data_abnormal) if normalize else data_abnormal
        self.data_normal = self.normalize(data_normal) if normalize else data_normal
        # self.seq_len = data_abnormal.size(2)
        
        #Estimates distribution parameters of deltas (Gaussian) from normalized data
        # original_deltas = data[:, -1] - data[:, 0]
        # self.original_deltas = original_deltas
        # self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min() 
        # deltas = self.data[:, -1] - self.data[:, 0]
        # self.deltas = deltas
        # self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        # self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self):
        return max(len(self.data_normal), len(self.data_abnormal))

    def __getitem__(self, idx):
        return {"Normal": self.data_normal[idx], "Abnormal": self.data_abnormal[:,0][idx]}

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min())/(x.max() - x.min()) - 1)
    
    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x*self.max - x*self.min + self.max + self.min)
    
    def sample_deltas(self, number):
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std
    
    def normalize_deltas(self, x):
        return ((self.delta_max - self.delta_min) * (x - self.or_delta_min)/(self.or_delta_max - self.or_delta_min) + self.delta_min)
    
if __name__ == "__main__":
    from pathlib import Path
    datapath = Path('./data/CaseI-Attacks without any change.csv')
    dataset = TsDataset(datapath)
    print(f"Successfuly loaded dataset with Normal:{dataset.data_normal.size()}|Abnormal:{dataset.data_abnormal.size()}.")
