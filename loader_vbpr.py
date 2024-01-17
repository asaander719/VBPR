import torch
from torch.utils.data import Dataset

class Load_Data(Dataset):
    def __init__(self, data_ori):
        self.data_ori = data_ori

    def __len__(self):
        return len(self.data_ori)

    def __getitem__(self, idx):
        data_ori = self.data_ori[idx]
        user_idx, top_idx, pos_bottom_idx, neg_bottom_idx = data_ori  
        return user_idx.long(), top_idx.long(), pos_bottom_idx.long(), neg_bottom_idx.long()