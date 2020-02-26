import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

class PUBG_dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # get one line in csv
        player_id = self.frame.ix[idx, 0]
        player_stats = torch.tensor(self.frame.ix[idx, [x for x in range(3, 27) if x != 15]].values.astype(np.int_))
        win_place_perc = torch.tensor(self.frame.ix[idx, 28])
        if self.transform:
            player_stats = self.transform(player_stats)
        sample = {
            "player_id": player_id, 
            "player_stats": player_stats,
            "win_place_perc": win_place_perc
        }
        return sample

def get_dataset(csv_file, train_dataset_size_ratio, batch_size):
    dataset = PUBG_dataset(csv_file)
    #  `torch.utils.data.random_split` meets server problem and lead to CRASH
    # see also:
    # - a denied fix PR for this problem: https://github.com/pytorch/pytorch/pull/9237 
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor((1-train_dataset_size_ratio) * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    print("load dataset: train dataset: {}, test dataset: {}.".format(len(train_loader)*batch_size, len(test_loader)*batch_size))
    return (train_loader, test_loader)
   
if __name__ == "__main__":
    csv_file = 'D:\\PUBGdata\\pubg-finish-placement-prediction\\train_V2.csv'
    train_dataset_size_ratio = 0.9
    batch_size = 16
    train_loader, test_loader = get_dataset(csv_file, train_dataset_size_ratio, batch_size)

    for data in train_loader:
        stats, prec = data['player_stats'], data['win_place_perc']
        #print(stats, prec)
        # test anything you want

    for i, data in enumerate(test_loader):
        stats, prec = data['player_stats'], data['win_place_perc']
        # test anything you want
    
    
