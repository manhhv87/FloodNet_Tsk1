import os
import numpy as np
from sklearn.model_selection import train_test_split


def labeled_ds(base_data_dir='/content/FloodNet/', train_size=0.8):
    flooded_root = os.path.join(base_data_dir, 'Train/Labeled/Flooded/image/')
    flooded_files = [flooded_root + name for name in os.listdir(flooded_root)]
    flooded_y = [1]*len(flooded_files)

    non_flooded_root = os.path.join(
        base_data_dir, 'Train/Labeled/Non-Flooded/image/')
    non_flooded_files = [non_flooded_root +
                         name for name in os.listdir(non_flooded_root)]
    non_flooded_y = [0]*len(non_flooded_files)

    X = flooded_files + non_flooded_files
    y = flooded_y + non_flooded_y

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=train_size, stratify=y)

    return X_train, X_valid, y_train, y_valid


def unlabeled_ds(base_data_dir='/content/FloodNet/'):
    unlabeled_root = os.path.join(base_data_dir, 'Train/Unlabeled/image/')
    unlabeled_file_names = os.listdir(unlabeled_root)
    unlabeled_file_names = list(
        map(lambda x: unlabeled_root+x, unlabeled_file_names))

    # Adding validation set to unlabeled data
    for ds in ['Validation', 'Test']:
        # data_root_ = os.path.join(base_data_dir, ds, "/image/")
        data_root_ = f"{base_data_dir}{ds}/image/"
        data_file_names_ = os.listdir(data_root_)
        unlabeled_file_names += list(map(lambda x: data_root_ +
                                     x, data_file_names_))

    unlabeled_file_names.sort()

    return unlabeled_file_names
