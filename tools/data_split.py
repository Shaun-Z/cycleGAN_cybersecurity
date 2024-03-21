# %%
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path

# %%
data_dir = os.path.join('.', 'data')
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# %%
df_lis = [pd.read_csv(file) for file in csv_files]

# %%
df = df_lis[0]

# %%
label_col = df.columns[-1]

# %%
data_save_path = Path(data_dir) / 'Casel_attack'
os.makedirs(data_save_path, exist_ok=True)

# %%
for col_name in df.columns[:-1]:
    # Create a new DataFrame containing the current and last columns
    new_df = df[[col_name, label_col]]
    # Save the new DataFrame as a CSV file with column names
    new_df.to_csv(data_save_path/f'{col_name}.csv', index=False)


