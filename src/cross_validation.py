"""Cross-validation classes."""

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from cross_validation_base import CrossValidationBase
from idrnd_metrics import compute_err


class CrossValidation(CrossValidationBase):

    def get_metrics(self, ytrue, ypred):
        """Calculation of dataframe with metrics."""

        ytrue2 = ytrue.squeeze()
        metrics = {}
        metrics['EER'] = compute_err(ypred[ytrue2 == 1], ypred[ytrue2 == 0])[0]
        metrics['EER2'] = eer_metric(ytrue2, ypred)
        metrics['roc_auc'] = roc_auc_score(ytrue2, ypred)
        metrics['logloss'] = log_loss(ytrue2, ypred)

        return metrics


def eer_metric_baseline(ytrue, ypred):
    """EER from baseline soution"""
    return compute_err(ypred[ytrue.squeeze() == 1],
                       ypred[ytrue.squeeze() == 0])[0]


def eer_metric(ytrue, ypred):
    """EER metric from https://yangcha.github.io/EER-ROC/"""

    fpr, tpr, thresholds = roc_curve(ytrue, ypred)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)  # optional, not needed here

    return eer*100
