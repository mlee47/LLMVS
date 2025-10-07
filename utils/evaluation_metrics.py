# -*- coding: utf-8 -*-
import torch
from torchmetrics.regression import SpearmanCorrCoef, KendallRankCorrCoef
import pdb
import csv
from collections import Counter
from scipy import stats
import numpy as np
import torch

# 텐서 출력 옵션 변경: 생략 없이 전체 출력
torch.set_printoptions(threshold=torch.inf)

def evaluate_summary(predicted_summary, user_summary, video_name, pred_score, eval_data):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray gt_summary: The user defined ground truth summaries (or summary).
    """
    y_pred2=predicted_summary
    y_true2=user_summary.mean(axis=0)

    if eval_data == 'summe':
        curr_rho_coeff = stats.spearmanr(y_pred2.cpu(), y_true2.cpu())[0]
        curr_tau_coeff = stats.kendalltau(stats.rankdata(-np.array(y_pred2.cpu())), stats.rankdata(-np.array(y_true2.cpu())))[0]
        return curr_tau_coeff, curr_rho_coeff

    else:
        with open('TVSum/ydata-tvsum50-anno.tsv') as annot_file:
            annot = list(csv.reader(annot_file, delimiter="\t"))
        annotation_length = list(Counter(np.array(annot)[:, 0]).values())
        user_scores = []
        for idx in range(1,51):
            init = (idx - 1) * annotation_length[idx-1]
            till = idx * annotation_length[idx-1]
            user_score = []
            for row in annot[init:till]:
                curr_user_score = row[2].split(",")
                curr_user_score = np.array([float(num) for num in curr_user_score])
                curr_user_score = curr_user_score / curr_user_score.max(initial=-1)
                curr_user_score = curr_user_score[::15]

                user_score.append(curr_user_score)
            user_scores.append(user_score)

        user = int(video_name.split("_")[-1])

        curr_user_score = user_scores[user-1]

        tmp_rho_coeff = []
        tmp_tau_coeff = []
        for annotation in range(len(curr_user_score)):
            true_user_score = curr_user_score[annotation]

            curr_rho_coeff = stats.spearmanr(pred_score.cpu(), true_user_score)[0]
            curr_tau_coeff = stats.kendalltau(stats.rankdata(-np.array(pred_score.cpu())), stats.rankdata(-np.array(true_user_score)))[0]

            tmp_rho_coeff.append(curr_rho_coeff)
            tmp_tau_coeff.append(curr_tau_coeff)
        pS = np.mean(tmp_rho_coeff)
        kT = np.mean(tmp_tau_coeff)
        return kT, pS