# ------------------------------------------------------------------------------
# --coding='utf-8'--
# if you have any problem， please contact zhai（wzhai2@uh.edu）
# ------------------------------------------------------------------------------

# this file is aimed for convert csv file to h5 file
import os
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

current_directory = os.getcwd()
data_directory = os.path.join(current_directory, 'simulatedata')
os.makedirs(data_directory, exist_ok=True)
files = os.listdir(data_directory)
csv_files = [f for f in files if f.endswith('.csv')]

if len(csv_files) == 0:
    print("No CSV files found in the current directory.")
else:
    for csv_file in csv_files:
        csv_file_path = os.path.join(data_directory, csv_file)
        print(f"Processing {csv_file}...")

        train_df = pd.read_csv(csv_file_path)
        e = np.array(train_df['status'], dtype=np.int32)
        t = np.array(train_df['time'], dtype=np.float32)
        x = np.array(train_df.drop([ 'status', 'time'], axis=1), dtype=np.float32)
        e = np.nan_to_num(e, nan=0)
        t = np.nan_to_num(t, nan=0)
        x = np.nan_to_num(x, nan=0)

        e_train, e_test = train_test_split(e, test_size=0.2, random_state=42)
        t_train, t_test = train_test_split(t, test_size=0.2, random_state=42)
        x_train, x_test = train_test_split(x, test_size=0.2, random_state=42)

        file_directory = os.path.join(data_directory, csv_file.replace('.csv', ''))
        os.makedirs(file_directory, exist_ok=True)

        file_path =  f'{csv_file.replace(".csv", "_data.h5")}'
        h5file = h5py.File(file_path, 'w')

        group_train = h5file.create_group('train')
        group_test = h5file.create_group('test')

        group_train.create_dataset('e', data=e_train)
        group_train.create_dataset('t', data=t_train)
        group_train.create_dataset('x', data=x_train)

        group_test.create_dataset('e', data=e_test)
        group_test.create_dataset('t', data=t_test)
        group_test.create_dataset('x', data=x_test)

        h5file.close()

print("Processing completed.")
