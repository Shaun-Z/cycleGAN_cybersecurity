import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA

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
    
class TsFullDataset(Dataset):
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

        data_normal = torch.from_numpy(group_normal.values).permute(1, 0).unsqueeze(2).float()
        data_abnormal = torch.from_numpy(np.array([group[1] for group in groups_abnormal])).permute(2, 1, 0).float()

        self.normal_len, self.abnormal_len = data_normal.size(1), data_abnormal.size(1)
        self.min_len = min(self.normal_len, self.abnormal_len)

        data_normal = data_normal[:, :self.min_len, :].repeat(1, 1, data_abnormal.size(2))  # Truncate the longer sequence. Repeat normal data to match abnormal data's feature length
        data_normal = data_normal + torch.randn_like(data_normal) * 0.1  # Add noise
        data_abnormal = data_abnormal[:, :self.min_len, :]

        self.data_abnormal = self.normalize(data_abnormal) if normalize else data_abnormal
        self.data_normal = self.normalize(data_normal) if normalize else data_normal

    def __len__(self):
        return max(len(self.data_normal), len(self.data_abnormal))

    def __getitem__(self, idx):
        return {"Normal": self.data_normal[idx], "Abnormal": self.data_abnormal[idx]}

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

class NewTsDataset(Dataset):
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

        data_normal = torch.from_numpy(group_normal.values).permute(1, 0).float().view(33, 1, -1) # Batch: places; Channel: Attack types; Feature: Time-series data

        data_abnormal = torch.from_numpy(np.array([group[1] for group in groups_abnormal])).permute(2, 0, 1).float()

        len1, len2 = data_normal.size(2), data_abnormal.size(2)

        min_len = min(len1, len2)

        data_normal = data_normal[:, :, :min_len]  # Truncate the longer sequence
        data_abnormal = data_abnormal[:, :1, :min_len]

        self.data_abnormal = self.normalize(data_abnormal) if normalize else data_abnormal
        self.data_normal = self.normalize(data_normal) if normalize else data_normal

    def __len__(self):
        return max(len(self.data_normal), len(self.data_abnormal))

    def __getitem__(self, idx):
        return {"Normal": self.data_normal[idx], "Abnormal": self.data_abnormal[idx]}

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
    
class TsSelfNoiseDataset(Dataset):
    """Time series dataset."""
    def __init__(self, csv_file, normalize=True, noise_type='gaussian'):
        """
        Args:
            csv_file (string): path to csv file
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = pd.read_csv(csv_file)
        df = df.set_index('Label').fillna(0)

        groups = df.groupby(df.index, sort=False)
        group_normal = groups.get_group('Normal')

        data_normal = torch.from_numpy(group_normal.values).permute(1, 0).unsqueeze(2).float()
        if noise_type == 'gaussian':
            data_noisy = data_normal + torch.randn_like(data_normal) * 0.1  # Add noise
        else:
            seasonal_freqs = [24, 365]  # Daily and yearly cycles
            arima_order = (5, 1, 0)  # ARIMA model order
            trend_degree = 2  # Polynomial trend degree
            data_noisy = self.generate_synthetic_data(data_normal, seasonal_freqs, arima_order, trend_degree)

        self.data_len = data_normal.size(1)

        self.data_normal = self.normalize(data_normal) if normalize else data_normal
        self.data_noisy = self.normalize(data_noisy) if normalize else data_noisy

    def generate_synthetic_data(self, normal_data, seasonal_freqs, arima_order, trend_degree):
        batch_size, seq_len, _ = normal_data.shape
        t = np.arange(seq_len)

        synthetic_data = np.zeros_like(normal_data)

        for i in range(batch_size):
            # Extract single sequence from the batch
            single_sequence = normal_data[i, :, 0]

            # Seasonal component
            seasonal_component = sum(np.sin(2 * np.pi * t / freq) for freq in seasonal_freqs)

            # Stochastic component
            single_sequence_np = single_sequence.cpu().detach().numpy()  # Convert to NumPy array
            model = ARIMA(single_sequence_np, order=arima_order)
            stochastic_component = model.fit().resid

            # Trend component
            trend_component = np.polyval(np.polyfit(t, single_sequence, trend_degree), t)

            # Combine components
            synthetic_sequence = single_sequence + seasonal_component + stochastic_component + trend_component

            # Store the synthetic sequence back in the synthetic_data array
            synthetic_data[i, :, 0] = synthetic_sequence

        return torch.tensor(synthetic_data).float()

    def __len__(self):
        return len(self.data_normal)

    def __getitem__(self, idx):
        return {"Normal": self.data_normal[idx], "Noisy": self.data_noisy[idx]}

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
    
class TsFinalDataset(Dataset):
    """Time series dataset."""
    def __init__(self, csv_file, normalize=True, noise_type='none'):
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

        data_normal = torch.from_numpy(group_normal.values).unsqueeze(0).float()

        if noise_type == 'none':
            data_attacked = torch.from_numpy(np.array([group[1] for group in groups_abnormal])).float()
            data_normal = data_normal.repeat(data_attacked.size(0), 1, 1)
            # 计算重复次数
            repeat_times = data_normal.size(1) // data_attacked.size(1)
            # 周期复制 # 裁剪较长数据
            data_attacked = data_attacked.repeat(1, repeat_times + 1, 1)[:, :data_normal.size(1), :]
        elif noise_type == 'gaussian':
            data_attacked = data_normal + torch.randn_like(data_normal) * 0.1  # Add noise
        else:
            seasonal_freqs = [24, 365]  # Daily and yearly cycles
            arima_order = (5, 1, 0)  # ARIMA model order
            trend_degree = 2  # Polynomial trend degree
            data_attacked = self.generate_synthetic_data(data_normal, seasonal_freqs, arima_order, trend_degree)

        self.batch_size, self.data_len, self.feature_dim = data_normal.shape

        self.data_normal = self.normalize(data_normal) if normalize else data_normal
        self.data_attacked = self.normalize(data_attacked) if normalize else data_attacked

    def generate_synthetic_data(self, normal_data, seasonal_freqs, arima_order, trend_degree):
        batch_size, seq_len, _ = normal_data.shape
        t = np.arange(seq_len)

        synthetic_data = np.zeros_like(normal_data)

        for i in range(batch_size):
            # Extract single sequence from the batch
            single_sequence = normal_data[i, :, 0]

            # Seasonal component
            seasonal_component = sum(np.sin(2 * np.pi * t / freq) for freq in seasonal_freqs)

            # Stochastic component
            single_sequence_np = single_sequence.cpu().detach().numpy()  # Convert to NumPy array
            model = ARIMA(single_sequence_np, order=arima_order)
            stochastic_component = model.fit().resid

            # Trend component
            trend_component = np.polyval(np.polyfit(t, single_sequence, trend_degree), t)

            # Combine components
            synthetic_sequence = single_sequence + seasonal_component + stochastic_component + trend_component

            # Store the synthetic sequence back in the synthetic_data array
            synthetic_data[i, :, 0] = synthetic_sequence

        return torch.tensor(synthetic_data).float()

    def __len__(self):
        return len(self.data_normal)

    def __getitem__(self, idx):
        return {"Normal": self.data_normal[idx], "Attacked": self.data_attacked[idx]}

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
    
class TsNewTestDataset(Dataset):
    """Time series dataset."""
    def __init__(self, csv_file, normalize=True, noise_type='none'):
        """
        Args:
            csv_file (string): path to csv file
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = pd.read_csv(csv_file)
        df = df.set_index('Label').fillna(0)

        data_all = torch.from_numpy(df.values).unsqueeze(0).float()

        groups = df.groupby(df.index, sort=False)
        group_normal = groups.get_group('Normal')
        group_normal_RE = groups.get_group('NormalafterIntegrationofRE')
        group_attacked = groups.get_group('AttacksBeforeIntegration')
        group_attacked_RE = groups.get_group('AttackafterIntegration')

        data_normal = torch.from_numpy(group_normal.values).unsqueeze(0).float()
        data_normal_RE = torch.from_numpy(group_normal_RE.values).unsqueeze(0).float()
        data_attacked = torch.from_numpy(group_attacked.values).unsqueeze(0).float()
        data_attacked_RE = torch.from_numpy(group_attacked_RE.values).unsqueeze(0).float()

        self.batch_size, self.data_len, self.feature_dim = data_normal.shape

        self.data_all = self.normalize(data_all) if normalize else data_all
        self.data_normal = self.normalize(data_normal) if normalize else data_normal
        self.data_normal_RE = self.normalize(data_normal_RE) if normalize else data_normal_RE
        self.data_attacked = self.normalize(data_attacked) if normalize else data_attacked
        self.data_attacked_RE = self.normalize(data_attacked_RE) if normalize else data_attacked_RE

    # def __len__(self):
    #     return len(self.data_normal)

    # def __getitem__(self, idx):
    #     return {"Normal": self.data_normal[idx], "Attacked": self.data_attacked[idx]}

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min())/(x.max() - x.min()) - 1)
    
    # def denormalize(self, x):
    #     """Revert [-1,1] normalization"""
    #     if not hasattr(self, 'max') or not hasattr(self, 'min'):
    #         raise Exception("You are calling denormalize, but the input was not normalized")
    #     return 0.5 * (x*self.max - x*self.min + self.max + self.min)
    
    
class TsModifiedDataset(Dataset):
    """Time series dataset."""
    def __init__(self, csv_file, normalize=True, noise_type='none'):
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

        data_normal = torch.from_numpy(group_normal.values).unsqueeze(0).float()

        if noise_type == 'none':
            data_attacked = torch.from_numpy(np.array([group[1] for group in groups_abnormal])).float()
            # data_normal = data_normal.repeat(data_attacked.size(0), 1, 1)
            data_normal = data_normal.repeat(data_attacked.size(0), 1, 1)[:, :1973, :]
            # 计算重复次数
            # repeat_times = data_normal.size(1) // data_attacked.size(1)
            repeat_times = 1973 // data_attacked.size(1)
            # 周期复制 # 裁剪较长数据
            # data_attacked = data_attacked.repeat(1, repeat_times + 1, 1)[:, :data_normal.size(1), :]
            data_attacked = data_attacked.repeat(1, repeat_times + 1, 1)[:, :1973, :]
        elif noise_type == 'gaussian':
            data_attacked = data_normal + torch.randn_like(data_normal) * 0.1  # Add noise
            data_normal, data_attacked = data_normal[:, :1973, :], data_attacked[:, :1973, :]
        else:
            seasonal_freqs = [24, 365]  # Daily and yearly cycles
            arima_order = (5, 1, 0)  # ARIMA model order
            trend_degree = 2  # Polynomial trend degree
            data_attacked = self.generate_synthetic_data(data_normal, seasonal_freqs, arima_order, trend_degree)
            data_normal, data_attacked = data_normal[:, :1973, :], data_attacked[:, :1973, :]

        self.batch_size, self.data_len, self.feature_dim = data_normal.shape

        data_normal = data_normal.permute(2, 0, 1)
        data_attacked = data_attacked.permute(2, 0, 1)

        self.data_normal = self.normalize(data_normal) if normalize else data_normal
        self.data_attacked = self.normalize(data_attacked) if normalize else data_attacked

    def generate_synthetic_data(self, normal_data, seasonal_freqs, arima_order, trend_degree):
        batch_size, seq_len, _ = normal_data.shape
        t = np.arange(seq_len)

        synthetic_data = np.zeros_like(normal_data)

        for i in range(batch_size):
            # Extract single sequence from the batch
            single_sequence = normal_data[i, :, 0]

            # Seasonal component
            seasonal_component = sum(np.sin(2 * np.pi * t / freq) for freq in seasonal_freqs)

            # Stochastic component
            single_sequence_np = single_sequence.cpu().detach().numpy()  # Convert to NumPy array
            model = ARIMA(single_sequence_np, order=arima_order)
            stochastic_component = model.fit().resid

            # Trend component
            trend_component = np.polyval(np.polyfit(t, single_sequence, trend_degree), t)

            # Combine components
            synthetic_sequence = single_sequence + seasonal_component + stochastic_component + trend_component

            # Store the synthetic sequence back in the synthetic_data array
            synthetic_data[i, :, 0] = synthetic_sequence

        return torch.tensor(synthetic_data).float()

    def __len__(self):
        return len(self.data_normal)

    def __getitem__(self, idx):
        return {"Normal": self.data_normal[idx], "Attacked": self.data_attacked[idx]}

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
    # datapath = Path('./data/CaseII.csv')

    dataset = TsFinalDataset(datapath, noise_type='none')
    print(dataset.max, dataset.min)
    print(f"Successfuly loaded dataset with Normal:{dataset.data_normal.size()}|Attacked:{dataset.data_attacked.size()}.")
    import matplotlib.pyplot as plt
    # Plotting the first time series in data_normal
    plt.plot(dataset.data_normal[0,:,0])
    plt.plot(dataset.data_attacked[0,:,0])
    plt.plot(dataset.data_normal[1,:,0])
    plt.plot(dataset.data_attacked[1,:,0])
    plt.plot(dataset.data_normal[2,:,0])
    plt.plot(dataset.data_attacked[2,:,0])
    plt.legend(['Normal0', 'Attacked0', 'Normal1', 'Attacked1', 'Normal2', 'Attacked2'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series')
    plt.show()

    print((dataset.data_attacked[3,:,0] - dataset.data_attacked[4,:,0]).sum())

    # dataset = TsModifiedDataset(datapath, noise_type='other')
    # print(f"Successfuly loaded dataset with Normal:{dataset.data_normal.size()}|Attacked:{dataset.data_attacked.size()}.")
    # import matplotlib.pyplot as plt
    # # Plotting the first time series in data_normal
    # plt.plot(dataset.data_normal[0,0,:])
    # plt.plot(dataset.data_attacked[0,0,:])
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Time Series')
    # plt.show()

    # dataset = TsDataset(datapath)
    # print(f"Successfuly loaded dataset with Normal:{dataset.data_normal.size()}|Attacked:{dataset.data_abnormal.size()}.")

    # import matplotlib.pyplot as plt
    # # Plotting the first time series in data_normal
    # plt.plot(dataset.data_normal[0,:])
    # plt.plot(dataset.data_abnormal[0,0,:])
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Time Series')
    # plt.show()