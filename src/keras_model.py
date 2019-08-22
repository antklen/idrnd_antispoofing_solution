"""
Wrapper for training keras models.
"""

import logging

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from keras_log_callbacks import LoggingCallback


class KerasModel():

    """Wrapper for training keras models."""

    def __init__(self, nn_model, num_epochs=20, validation_size=0.1,
                 batch_size=32, batch_size_pred=128,
                 patience=3, verbose=1,
                 reduce_lr_params=None, restore_best=True,
                 class_weights=None, random_state=42, num_workers=4,
                 use_multiprocessing=True, max_queue_size=10,
                 logger=None):

        self.nn_model = nn_model
        self.num_epochs = num_epochs
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.batch_size_pred = batch_size_pred
        self.patience = patience
        self.verbose = verbose
        self.reduce_lr_params = reduce_lr_params
        self.restore_best = restore_best
        self.class_weights = class_weights
        self.random_state = random_state
        self.num_workers = num_workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger


    def train(self, Xtrain, ytrain, Xval, yval,
              Xtest=None, weights_train=None):
        """Train model and predict on validation and test data."""

        callbacks = self.get_callbacks()

        if self.validation_size > 0:
            Xtrain, Xval_inner, ytrain, yval_inner = train_test_split(
                Xtrain, ytrain, test_size=self.validation_size,
                random_state=self.random_state)
            validation_data = (Xval_inner, yval_inner)
        else:
            validation_data = None

        model = self.nn_model()
        model.fit(Xtrain, ytrain,
                  validation_data=validation_data,
                  epochs=self.num_epochs, batch_size=self.batch_size,
                  sample_weight=weights_train, class_weight=self.class_weights,
                  callbacks=callbacks, verbose=self.verbose)

        pred, pred_test = self.predict(model, Xval, Xtest)

        return model, pred, pred_test


    def train_generator(self, meta_train, meta_val, meta_test,
                        path, path_test, preprocessing_fn,
                        data_generator, data_generator_params,
                        target_cols='target'):
        """
        Train model and predict on validation and test data
        using generators.
        """

        callbacks = self.get_callbacks()

        if self.validation_size > 0:
            meta_train, meta_val_inner = train_test_split(
                meta_train, test_size=self.validation_size,
                random_state=self.random_state)
            valid_inner_generator = data_generator(
                path=path,
                IDs=meta_val_inner.index.tolist(),
                labels=meta_val_inner[target_cols],
                preprocessing_fn=preprocessing_fn,
                shuffle=False, batch_size=self.batch_size_pred,
                **data_generator_params)
        else:
            valid_inner_generator = None

        train_generator = data_generator(
            path=path,
            IDs=meta_train.index.tolist(),
            labels=meta_train[target_cols],
            preprocessing_fn=preprocessing_fn,
            shuffle=True, batch_size=self.batch_size,
            **data_generator_params)

        valid_generator = data_generator(
            path=path,
            IDs=meta_val.index.tolist(),
            labels=meta_val[target_cols],
            preprocessing_fn=preprocessing_fn,
            shuffle=False, batch_size=self.batch_size_pred,
            **data_generator_params)

        test_generator = data_generator(
            path=path_test,
            IDs=meta_test.index.tolist(),
            preprocessing_fn=preprocessing_fn,
            shuffle=False, batch_size=self.batch_size_pred,
            **data_generator_params)

        model = self.nn_model()
        model.fit_generator(train_generator, epochs=self.num_epochs,
                            callbacks=callbacks, verbose=self.verbose,
                            validation_data=valid_inner_generator,
                            class_weight=self.class_weights,
                            workers=self.num_workers,
                            use_multiprocessing=self.use_multiprocessing,
                            max_queue_size=self.max_queue_size)

        pred, pred_test = self.predict_generator(
            model, valid_generator, test_generator)

        return model, pred, pred_test


    def get_callbacks(self):
        """Prepare callbacks for model training."""

        early_stopping = EarlyStopping(
            monitor='val_loss', mode='auto',
            verbose=1, patience=self.patience,
            restore_best_weights=self.restore_best)
        callbacks = [early_stopping]
        if self.reduce_lr_params is not None:
            reduce_lr = ReduceLROnPlateau(**self.reduce_lr_params)
            callbacks.append(reduce_lr)

        callbacks.append(LoggingCallback(self.logger.info))

        return callbacks


    def predict(self, model, Xval, Xtest):
        """Make predictions with single set of weights."""

        pred = model.predict(
            Xval, batch_size=self.batch_size_pred, verbose=0).squeeze()
        pred_test = model.predict(
            Xtest, batch_size=self.batch_size_pred, verbose=0).squeeze()

        return pred, pred_test


    def predict_generator(self, model, valid_generator, test_generator):
        """Make predictions with single set of weights using generators."""

        pred = model.predict_generator(
            valid_generator, workers=self.num_workers,
            use_multiprocessing=self.use_multiprocessing, verbose=0).squeeze()
        pred_test = model.predict_generator(
            test_generator, workers=self.num_workers,
            use_multiprocessing=self.use_multiprocessing, verbose=0).squeeze()

        return pred, pred_test
