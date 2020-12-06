# !/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import numpy as np




def inference_fn(test_loader, model, device):

    model.eval()
    preds = []

    for step, (cont_x, cate_x) in enumerate(test_loader):

        cont_x,  cate_x = cont_x.to(device), cate_x.to(device)

        with torch.no_grad():
            pred = model(cont_x, cate_x)

        preds.append(pred.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds