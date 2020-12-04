#!/usr/bin/python
# -*- coding: utf-8 -*-
immport torch
import torch.nn as nn




class TabularNN(nn.Module):
    def __init__(self, cfg: CFG) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
                          nn.Linear(len(cfg.num_features), cfg.hidden_size),
                          nn.BatchNorm1d(cfg.hidden_size),
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.Linear(cfg.hidden_size, cfg.hidden_size),
                          nn.BatchNorm1d(cfg.hidden_size),
                          nn.Dropout(cfg.dropout),
                          nn.PReLU(),
                          nn.Linear(cfg.hidden_size, len(cfg.target_cols)),
                          )

    def forward(self, cont_x: torch.FloatTensor, cate_x: torch.LongTensor) -> torch.FloatTensor:
        # no use of cate_x yet
        x = self.mlp(cont_x)
        return x
