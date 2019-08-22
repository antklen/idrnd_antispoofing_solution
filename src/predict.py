"""Make test set predictions."""

import os
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras.models import load_model

import data_generator
from data_generator import Generator


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
tf.logging.set_verbosity(tf.logging.ERROR)


dataset_dir = "."
CONFIG_PATH = 'config.yml'
MODELS_PATH = '../models'
NUM_FOLDS = 3


eval_protocol_path = "protocol_test.txt"
eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
eval_protocol.columns = ['path', 'key']
print(eval_protocol.shape)
print(eval_protocol.sample(5).head())

with open(CONFIG_PATH, 'r') as f:
    configs = yaml.load(f.read())


def make_cv_predictions(model_id, config, dataset_dir,
                        eval_protocol, n_folds=5):

    function = getattr(data_generator, config['preprocess_fn'])
    preprocess_fn = partial(function, **config['preprocess'])
    generator = Generator(path=dataset_dir,
                          IDs=eval_protocol.path.tolist(),
                          preprocessing_fn=preprocess_fn,
                          shuffle=False, batch_size=128,
                          **config['generator'])

    preds = []
    for i in range(n_folds):
        model = load_model(os.path.join(
            MODELS_PATH, 'model_{}_{}.h5'.format(model_id, i)))
        preds.append(model.predict_generator(
            generator, use_multiprocessing=True, workers=6, verbose=0))

    return np.array(preds).mean(axis=0)


preds = []
for model_id in configs.keys():
    config = configs[model_id]
    preds.append(make_cv_predictions(model_id, config, dataset_dir,
                                     eval_protocol, n_folds=NUM_FOLDS))


eval_protocol['score'] = np.array(preds).mean(axis=0)

eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
print(eval_protocol.sample(5).head())
