"""Script with metrics from baseline solution from hosts."""

# Copyright 2019. DSPLabs. All Rights Reserved.

import numpy as np


def compute_frr_far(tar, imp):
    pt = np.concatenate((np.ones(len(tar)), np.zeros(len(imp))), axis=0)
    pi = np.concatenate((np.zeros(len(tar)), np.ones(len(imp))), axis=0)

    tar_imp = np.hstack((tar, imp))
    arg_sort = np.argsort(tar_imp)

    pt = pt[arg_sort]
    pi = pi[arg_sort]
    tar_imp = tar_imp[arg_sort]

    fr = np.zeros(pt.shape[0])
    fa = np.zeros(pi.shape[0])

    for i in range(1, len(pt)):
        fr[i] = fr[i - 1] + pt[i - 1]

    for i in range(len(pt) - 2, -1, -1):
        fa[i] = fa[i + 1] + pi[i]

    frr = fr / len(tar)
    far = fa / len(imp)

    return tar_imp, frr, far


def compute_err(tar, imp):
    tar_imp, fr, fa = compute_frr_far(tar, imp)

    index_min = np.argmin(np.abs(fr - fa))
    eer = 100.0 * np.mean((fr[index_min], fa[index_min]))
    threshold = tar_imp[index_min]

    return eer, threshold


def compute_min_c(pt, tar, imp, c_miss=1, c_fa=1):
    tar_imp, fnr, fpr = compute_frr_far(tar, imp)

    beta = c_fa * (1 - pt) / (c_miss * pt)
    log_beta = np.log(beta)
    act_c = fnr + beta * fpr
    index_min = np.argmin(act_c)
    min_c = act_c[index_min]
    threshold = tar_imp[index_min]

    return min_c, threshold, log_beta


def compute_act_c(pt, tar, imp, c_miss=1, c_fa=1):
    beta = c_fa * (1 - pt) / (c_miss * pt)
    log_beta = np.log(beta)

    f_tar = list(filter(lambda t: t < log_beta, tar))
    f_imp = list(filter(lambda i: i > log_beta, imp))

    fnr = len(f_tar) / len(tar)
    fpr = len(f_imp) / len(imp)

    act_c = fnr + beta * fpr

    return act_c, fpr, fnr


def compute_llr_c(tar, imp):
    sum_tar = np.sum([np.log(1. + 1. / np.exp(score)) for score in tar])
    sum_imp = np.sum([np.log(1. + np.exp(score)) for score in imp])

    c_llr = 1 / (2 * np.log(2)) * (sum_tar / len(tar) + sum_imp / len(imp))

    return c_llr


def get_eer(tar, imp):
    return compute_err(tar, imp)[0]


def get_min_c(p_target, tar, imp, c_miss=1, c_fa=1):
    if not hasattr(p_target, '__iter__'):
        p_target = [p_target]

    values = list(map(lambda pt: compute_min_c(pt, tar, imp, c_miss, c_fa)[0], p_target))

    return sum(values) / len(values)


def get_act_c(p_target, tar, imp, c_miss=1, c_fa=1):
    if not hasattr(p_target, '__iter__'):
        p_target = [p_target]

    values = list(map(lambda pt: compute_act_c(pt, tar, imp, c_miss, c_fa)[0], p_target))

    return sum(values) / len(values)


def get_llr_c(tar, imp):
    return compute_llr_c(tar, imp)


def get_fr_fa_at_threshold(tar, imp, threshold=0.5):
    fr = len(np.where(tar < threshold)[0])
    fa = len(np.where(imp > threshold)[0])
    fr = fr * 100. / len(tar)
    fa = fa * 100. / len(imp)
    return fr, fa


def get_acer_at_threshold(tar, imp, threshold=0.5):
    fr, fa = get_fr_fa_at_threshold(tar, imp, threshold=threshold)
    return (fr + fa) / 2.0


def get_bpcer_at_apcer(tar, imp, apcer=0.01):
    tar_imp, fr, fa = compute_frr_far(tar, imp)
    return 100.0 * fr[np.argmax(fa <= apcer)]
