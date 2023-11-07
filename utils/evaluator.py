import numpy as np
import logging
import sklearn.metrics as sklmetric
import tqdm
import json
from collections import defaultdict


def average_precision(truth, scores):
    if np.sum(truth > 0) > 0:
        # AUC sklmetric.roc_auc_score(truth, scores)
        a = sklmetric.average_precision_score(truth, scores)
        assert not np.isnan(a)
        return a
    else:
        return np.nan


def calc_roc_auc(truth, scores):
    if np.sum(truth > 0) > 0:
        a = sklmetric.roc_auc_score(truth, scores)
        assert not np.isnan(a)
        return a
    else:
        return np.nan


def tde_evaluator(pred, gt):
    '''
    :param pred:    dict includes attr_main (bz * num_attr), aff_main (bz * num_aff), tde_main (bz * num_attr * num_aff)
    :param gt:      dict includes gt_attr (bz * num_attr), gt_aff (bz * num_aff), gt_causal (bz * num_attr * num_aff)
    :return:
    '''

    many300 = np.array(json.load(open('./data/resources/causal_many_shot300.json')))

    num_attr = gt['attr'].shape[1]
    num_aff = gt['aff'].shape[1]

    ######## simple
    AP_simple = np.zeros((num_attr, num_aff))
    prob_diff = pred['tde_main']
    score_simple = np.where(
        np.tile(np.expand_dims(gt['aff'] == 1, 1), (1, num_attr, 1)) ^ (prob_diff > 0),
        0,
        np.abs(prob_diff)
    )
    for i in range(num_attr):
        for j in range(num_aff):
            AP_simple[i, j] = average_precision(
                gt['causal'][:, i, j],
                score_simple[:, i, j]
            )

    mAP_simple_many300 = np.nanmean(np.where(many300>0,AP_simple,np.nan))



    
    ######## merge
    score_attr = np.where(
        gt['attr'] == 1,
        pred['attr_main'],
        1 - pred['attr_main']
    )
    score_aff = np.where(
        gt['aff'] == 1,
        pred['aff_main'],
        1 - pred['aff_main']
    )
    score_merge = score_simple * \
                  np.tile(np.expand_dims(score_attr, 2), (1, 1, num_aff)) * \
                  np.tile(np.expand_dims(score_aff, 1), (1, num_attr, 1))

    AP_merge = np.zeros((num_attr, num_aff))
    AUC_merge = np.zeros((num_attr, num_aff))
    for i in range(num_attr):
        for j in range(num_aff):
            AP_merge[i, j] = average_precision(
                gt['causal'][:, i, j],
                score_merge[:, i, j]
            )
            AUC_merge[i, j] = calc_roc_auc(
                gt['causal'][:, i, j],
                score_merge[:, i, j]
            )
            
    mAP_merge_many300 = np.nanmean(np.where(many300>0,AP_merge,np.nan))

    ############################### ACC
    gt_attr_per_aff = np.transpose(gt['causal'], (0, 2, 1)).reshape(-1, num_attr)
    mask = gt_attr_per_aff.sum(-1) > 0

    gt_attr_per_aff = gt_attr_per_aff[mask]

    score_attr_per_aff = np.where(
        np.tile(np.expand_dims(gt['aff'] == 1, 1), (1, num_attr, 1)),
        prob_diff,
        - prob_diff
    )
    score_attr_per_aff = np.transpose(score_attr_per_aff, (0, 2, 1)).reshape(-1, num_attr)
    score_attr_per_aff = score_attr_per_aff[mask]

    idx = np.argsort(score_attr_per_aff, 1)[:, ::-1]

    top1_label = gt_attr_per_aff[
        np.tile(np.expand_dims(np.arange(len(gt_attr_per_aff)), 1), 1), idx[:, :1]]
    top1_acc = np.mean(top1_label.max(1))

    top5_label = gt_attr_per_aff[
        np.tile(np.expand_dims(np.arange(len(gt_attr_per_aff)), 1), 5), idx[:, :5]]
    top5_acc = np.mean(top5_label.max(1))


    return {
        'top300_ITE_mAP': mAP_simple_many300 * 100,
        'top300_abITE_mAP': mAP_merge_many300 * 100,
        'top1_acc_simple': top1_acc * 100,
        'top5_acc_simple': top5_acc * 100
    }


def mAP_evaluator(prediction, gt_attr, store_ap=None, return_vec=False):
    """prediction, gt_attr: (#instance, #category)
    return mAP(float)"""
    assert prediction.shape == gt_attr.shape

    assert not np.any(np.isnan(prediction)), str(np.sum(np.isnan(prediction)))
    assert not np.any(np.isnan(gt_attr)), str(np.sum(np.isnan(gt_attr)))

    ap = np.zeros((gt_attr.shape[1],))
    pos = np.zeros((gt_attr.shape[1],))  # num of positive sample

    for dim in range(gt_attr.shape[1]):
        # rescale ground truth to [-1, 1]

        gt = gt_attr[:, dim]
        mask = (gt >= 0)

        gt = 2 * gt[mask] - 1  # = 0.5 threshold
        est = prediction[mask, dim]

        ap[dim] = average_precision(gt, est)
        pos[dim] = np.sum(gt > 0)

    if store_ap is not None:
        import os
        assert not os.path.exists(store_ap + '.txt')
        with open(store_ap + '.txt', 'w') as f:
            for dim in range(gt_attr.shape[1]):
                f.write("Dim %d AP %f\n" % (dim, ap[dim]))

    if return_vec:
        return ap
    else:
        mAP = np.nanmean(ap)
        return mAP * 100

