"""Train all models."""

import os
from functools import partial

import pandas as pd
import tensorflow as tf
import yaml

import data_generator
import models
from create_logger import create_logger
from cross_validation import CrossValidation
from data_generator import Generator, generate_train_data
from keras_model import KerasModel


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)


PATH_TRAIN = '../data/train'
PATH_TEST = '../data/train'
PATH_TRAIN_META = '../data/meta_train.csv'
CONFIG_PATH = 'config.yml'
MODELS_PATH = '../models'
LOGS_PATH = '../logs'


meta_train = pd.read_csv(PATH_TRAIN_META).set_index('fname')
meta_test = meta_train.iloc[:100]  # placeholder for meta_test

with open(CONFIG_PATH, 'r') as f:
    configs = yaml.load(f.read())

log_file = os.path.join(LOGS_PATH, __file__.split('.')[0] + '.log')
logger = create_logger(logger_name=__name__, log_file=log_file)


def run(model_id):
    """Run experiment."""

    config = configs[model_id]
    logger.info('\n\n\ntrain model {}'.format(model_id))

    # prepare data
    if config['preprocess_fn'] is not None:
        function = getattr(data_generator, config['preprocess_fn'])
        preprocess_fn = partial(function, **config['preprocess'])
    else:
        preprocess_fn = None
    generator = Generator(path=PATH_TRAIN,
                          IDs=meta_train.index.tolist(),
                          labels=meta_train[['target']],
                          preprocessing_fn=preprocess_fn,
                          shuffle=False, batch_size=64,
                          **config['generator'])
    X, y = generate_train_data(generator, meta_train)
    logger.info('X shape: {}, y shape: {}'.format(X.shape, y.shape))

    # define model
    model_function = getattr(models, config['model_name'])
    nn_model = partial(model_function,
                       input_shape=(X.shape[1:]),
                       **config['model_params'])
    nn_model().summary(print_fn=logger.info)
    model = KerasModel(nn_model, logger=logger, **config['train'])

    # train and save model
    cross_val = CrossValidation(X=X, y=y, Xtest=X[:100],
                                logger=logger, **config['cv'])
    pred, pred_test, metrics, trained_models = cross_val.run_cv(model)

    for i, model in enumerate(trained_models):
        path = os.path.join(MODELS_PATH, 'model_{}_{}.h5'.format(model_id, i))
        model.save(path)


for model_id in configs.keys():
    run(model_id=model_id)
