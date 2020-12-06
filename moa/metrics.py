# !/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score




# Final result with 'cp_type'=='ctl_vehicle' data
train[target_cols] = oof

score = 0
for i in range(target.shape[1]):
    _score = log_loss(target[:,i], oof[:,i])
    score += _score / target.shape[1]
print(f"Seed Averaged CV score: {score}")

from sklearn.metrics import roc_auc_score

aucs = []
for i in range(target.shape[1]):
    aucs.append(roc_auc_score(y_true=target[:, i],
                              y_score=oof[:, i]))
print(f"Overall AUC : {np.mean(aucs)}")
print(f"Average CV : {np.mean(score)}")                                  