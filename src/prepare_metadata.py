"""Prepare metadata."""

import os
import pandas as pd

PATH_TRAIN = '../data/train'
PATH_TRAIN_META = '../data/meta_train.csv'

train_files = os.listdir(PATH_TRAIN)

meta_train = pd.DataFrame({'fname': train_files})
meta_train['target'] = meta_train.fname.map(
    lambda x: 1 if x.split('_')[0] == 'human' else 0)
print(meta_train.shape)
print(meta_train.head())

meta_train.to_csv(PATH_TRAIN_META, index=False)
