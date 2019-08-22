"""
Cross-validation base classes.
"""

import logging
import timeit
from abc import ABCMeta, abstractmethod

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


class CrossValidationBase(metaclass=ABCMeta):

    """Base cross-validation class."""

    def __init__(self, X, y, Xtest=None, groups=None, weights=None,
                 n_folds=5, cv_type='kfold', only_first_fold=False,
                 random_state=42, logger=None):

        self.X = X
        self.y = y
        self.Xtest = Xtest
        self.groups = groups
        self.weights = weights
        self.n_folds = n_folds
        self.cv_type = cv_type
        self.only_first_fold = only_first_fold
        self.random_state = random_state

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.init_fold_indexes(split_X=X, split_y=y)


    def init_fold_indexes(self, split_X, split_y):
        """Initialize indexes for each fold."""

        if self.cv_type == 'kfold':
            kf = KFold(self.n_folds, shuffle=True,
                       random_state=self.random_state)
        elif self.cv_type == 'stratified':
            kf = StratifiedKFold(self.n_folds, shuffle=True,
                                 random_state=self.random_state)
        elif self.cv_type == 'grouped':
            kf = GroupKFold(self.n_folds)
        elif self.cv_type == 'custom':
            self.init_custom_fold_indexes()
            return

        fold_indexes = []
        for (train_index, test_index) in kf.split(X=split_X, y=split_y,
                                                  groups=self.groups):
            fold_indexes.append({'train': train_index,
                                 'validation': test_index})
        self.fold_indexes = fold_indexes


    def init_custom_fold_indexes(self):
        """Custom strategy for train-validation split."""

        raise NotImplementedError('Custom CV strategy is not implemented.')


    def run_cv(self, model):
        """Run cross-validation loop."""

        pred_oof = self.init_oof_predictions()
        pred_test = []
        fold_metrics = []
        trained_models = []

        for fold_number, fold_index in enumerate(self.fold_indexes):

            self.logger.info('='*70)
            self.logger.info('Fold %d' % fold_number)
            start_time = timeit.default_timer()

            Xtrain, Xval, ytrain, yval, weights_train = \
                self.get_fold_values(fold_number)

            trained_model, pred_fold, pred_fold_test = model.train(
                Xtrain, ytrain, Xval, yval, self.Xtest, weights_train)

            trained_models.append(trained_model)
            pred_oof.iloc[fold_index['validation']] = pred_fold
            pred_test.append(pred_fold_test)
            fold_metrics.append(self.get_metrics(yval, pred_fold))
            self.logger.info('fold metrics')
            self.logger.info(fold_metrics[-1])
            self.logger.info('time elapsed {:.0f}s'.format(
                timeit.default_timer()-start_time))

            if self.only_first_fold:
                break

        metrics = self.get_final_metrics(self.y, pred_oof, fold_metrics)

        return pred_oof, pred_test, metrics, trained_models


    def init_oof_predictions(self):
        """Initialize Series/DataFrame for out-of-fold predictions."""

        if len(self.y.shape) == 1:
            pred = pd.Series(index=self.y.index)
        elif self.y.shape[1] == 1:
            pred = pd.Series(index=self.y.index)
        elif self.y.shape[1] > 1:
            pred = pd.DataFrame(index=self.y.index, columns=self.y.columns)

        return pred


    def get_fold_values(self, fold):
        """Get data for given fold."""

        fold_index = self.fold_indexes[fold]
        if isinstance(self.X, pd.DataFrame):
            Xtrain = self.X.iloc[fold_index['train']]
            Xval = self.X.iloc[fold_index['validation']]
        else:
            Xtrain = self.X[fold_index['train']]
            Xval = self.X[fold_index['validation']]
        if isinstance(self.y, pd.DataFrame) or isinstance(self.y, pd.Series):
            ytrain = self.y.iloc[fold_index['train']]
            yval = self.y.iloc[fold_index['validation']]
        else:
            ytrain = self.y[fold_index['train']]
            yval = self.y[fold_index['validation']]
        if self.weights is not None:
            weights_train = self.weights[fold_index['train']]
        else:
            weights_train = None

        return Xtrain, Xval, ytrain, yval, weights_train


    def get_final_metrics(self, ytrue, ypred, fold_metrics):
        """Combine per fold metrics with out-of-fold metrics"""

        metrics = pd.DataFrame(fold_metrics).T
        metrics.columns = metrics.columns.map(lambda x: 'Fold %d' % x)
        if not self.only_first_fold:
            metrics['Fold average'] = metrics.mean(axis=1)
            oof_metrics = self.get_metrics(ytrue.squeeze(), ypred)
            metrics['OOF'] = pd.Series(oof_metrics)
            self.logger.info('oof metrics')
            self.logger.info(oof_metrics)
        else:
            metrics = metrics.rename(columns={'Fold 0': 'OOF'})

        self.logger.info('='*70)
        self.logger.info('final metrics')
        self.logger.info(metrics)

        return metrics


    @abstractmethod
    def get_metrics(self, ytrue, ypred):
        """Calculate metrics"""
