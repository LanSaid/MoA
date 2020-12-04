# !/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset



class TrainDataset(Dataset):
    def __init__(self, df, num_features: np.ndarray, cat_features: np.ndarray, labels: np.ndarray) -> None:
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.cont_values)

    def __getitem__(self, idx: int) -> torch.FloatTensor, torch.LongTensor, torch.LongTensor:
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])
        label = torch.tensor(self.labels[idx]).float()
        
        return cont_x, cate_x, label
    

class TestDataset(Dataset):
    def __init__(self, df, num_features: np.ndarray, cat_features: np.ndarray) -> None:
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        
    def __len__(self) -> int:
        return len(self.cont_values)

    def __getitem__(self, idx: int) -> torch.FloatTensor, torch.LongTensor:
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])
        
        return cont_x, cate_x
