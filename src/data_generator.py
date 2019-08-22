import os

import librosa
import numpy as np
import pandas as pd
from keras.utils import Sequence
from scipy.io import wavfile
from tqdm import tqdm


class Generator(Sequence):
    """Keras data generator."""

    def __init__(self, path, IDs, labels=None,
                 batch_size=64, max_len=100000,
                 preprocessing_fn=None,
                 shuffle=True, expand_dims=False,
                 scaling='batch', trim=None,
                 repeat_short=False, padding=True):

        self.path = path
        self.IDs = IDs
        self.labels = labels
        self.batch_size = batch_size
        self.max_len = max_len
        self.preprocessing_fn = preprocessing_fn
        self.shuffle = shuffle
        self.expand_dims = expand_dims
        self.scaling = scaling
        self.trim = trim
        self.repeat_short = repeat_short
        self.padding = padding

        self.indexes = np.arange(len(self.IDs))
        self.on_epoch_end()

    def __len__(self):

        return int(np.ceil(len(self.IDs) / self.batch_size))

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        IDs = [self.IDs[idx] for idx in indexes]

        X = []

        for ID in IDs:

            file_path = os.path.join(self.path, ID)
            sample_rate, x = wavfile.read(file_path)
            if self.trim is not None:
                x, _ = librosa.effects.trim(x.astype('float'),
                                            top_db=self.trim)

            if self.padding:
                x = self.pad(x)
            if self.preprocessing_fn is not None:
                x = self.preprocessing_fn(x)

            X.append(x)

        X = np.array(X)
        if self.expand_dims:
            X = np.expand_dims(X, axis=-1)
        if self.scaling == 'batch':
            X = (X-X.mean())/X.std()

        if self.labels is not None:
            y = [self.labels.loc[ID] for ID in IDs]
            y = np.array(y)
            return X, y
        else:
            return X

    def pad(self, x):
        """Padding sequences to given max_len."""

        if self.repeat_short and len(x) < self.max_len:
            x = np.concatenate([x] * int(np.ceil(self.max_len/len(x))))
        if len(x) > self.max_len:
            max_offset = len(x) - self.max_len
            offset = np.random.randint(max_offset)
            x = x[offset:(self.max_len+offset)]
        elif self.max_len > len(x):
            x = np.pad(x, (0, self.max_len - len(x)), "constant")

        return x


def mel_spectrogram(x, sample_rate=44100, n_mels=128,
                    hop_length=512, n_fft=2048,
                    fmin=0, fmax=None):
    """Calculate melspectrogram for one sample."""

    spec = librosa.feature.melspectrogram(
        x.astype('float'), sr=sample_rate, n_mels=n_mels,
        hop_length=hop_length, n_fft=n_fft,
        fmin=fmin, fmax=fmax)
    spec = librosa.power_to_db(spec, ref=np.max)

    return spec.T


def constant_q_transform(x, sample_rate=44100, n_bins=84,
                         hop_length=512):
    """Calculate constant Q transform for one sample."""

    cqt = np.abs(librosa.cqt(
        x.astype('float'), sr=sample_rate,
        hop_length=hop_length, n_bins=n_bins))
    cqt = librosa.amplitude_to_db(cqt, ref=np.max)

    return cqt.T


def generate_train_data(generator, meta_train):
    """Preprocess all train data with generator."""

    print('preprocess train data..')
    X = []
    y = []
    for i in tqdm(range(len(generator))):
        x_i, y_i = generator[i]
        X.append(x_i)
        y.append(y_i)
    X = np.concatenate(X)
    y = pd.DataFrame(np.concatenate(y), index=meta_train.index)

    return X, y
