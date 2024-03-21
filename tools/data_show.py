# %%
import numpy as np
import pandas as pd

import os
import glob
from pathlib import Path

import matplotlib.cm as cm
from matplotlib import pyplot as plt

# %%
data_dir = os.path.join('.', 'data', 'Casel_attack')
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# %%
df_lis = [pd.read_csv(file) for file in csv_files]

# %%
# df = df_lis[0]
# # Creating a color map
# colors = cm.rainbow(np.linspace(0, 1, len(df.iloc[:, 1].unique())))
# # Plotting scatterplots with color labels
# for label, color in zip(df.iloc[:, 1].unique(), colors):
#     index = df.iloc[:, 1] == label
#     plt.scatter(df.index[index], df.iloc[:, 0][index], color=color, label=label)
#     # plt.plot(df.iloc[:, 0][index], color=color, label=label)
# # Set the title of the diagram to the file name
# plt.title(df.columns[0])
# plt.legend()
# plt.show()

path_to_save_image = Path(data_dir) / 'images'
os.makedirs(path_to_save_image, exist_ok=True)

for df in df_lis:
    # Create a new image and set its size
    plt.figure(figsize=(24, 12))
    # Creating a color map
    colors = cm.rainbow(np.linspace(0, 1, len(df.iloc[:, 1].unique())))
    # Plotting scatterplots with color labels
    for label, color in zip(df.iloc[:, 1].unique(), colors):
        index = df.iloc[:, 1] == label
        plt.scatter(df.index[index], df.iloc[:, 0][index], color=color, label=label)
        # plt.plot(df.iloc[:, 0][index], color=color, label=label)
    # Set the title of the diagram to the file name
    plt.title(df.columns[0])
    plt.legend()
    plt.savefig(path_to_save_image/f'{df.columns[0]}.png')
    # plt.show()
    plt.close()
