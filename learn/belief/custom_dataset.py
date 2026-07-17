import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from learn.belief.preprocess import split_data


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            data_name: str,
            flag: str,
            seed: int = 1,
    ) -> None:
        '''
        class for holding and loading data for torch dataLoader

        input
        -----
        datadir:str
            name of the folder with all training data
        name:str
            name of folder in datadir w/ specific training data
        flag:str
            which dataset to track (train,test,validation)
        seed:int
            seed
        '''
        self.seed = seed
        self.flag = flag
        self.data_title = os.path.join(data_dir,data_name)

        self.data_dir = data_dir
        self.data_name = data_name
        self.filelist = None
        self.folderlist = None
        self.update_filelist()

    def __len__(self):
        return len(self.filelist)
    
    def get_filelist(self):
        return self.filelist

    def get_folderlist(self):
        return self.folderlist

    def update_filelist(self) -> None:
        '''
        update file list using data split file, creates a list of  data and label file paires 
        for appropriate dataset based on self.flag
        '''
        filelist = []
        folderlist = []
        data_filepath = os.path.join(self.data_dir, self.data_name)

        # get the data ids from data split file
        filelist_path = os.path.join('data/datainfo', self.data_name, f'data_split_dict_{self.seed}.json')
        if not Path(filelist_path).exists():
            split_data('data/', name=self.data_name, seed=self.seed)

        time.sleep(5)

        with open(filelist_path, 'r') as file:
            seq_dict = json.load(file)
        
        data_list = seq_dict[self.flag]

        for data_idx in data_list:
            seq_filepath = os.path.join(data_filepath, str(data_idx))
            dataps = os.listdir(seq_filepath)
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            #append pairs of data and label file names ([x,y]) to total file list
            folderlist.append(seq_filepath)
            for datap in dataps:
                filelist.append(os.path.join(seq_filepath, datap))

        self.filelist = filelist
        self.folderlist = folderlist
    
    def __getitem__(
            self,
            idx: int,
    ):
        '''
        returns data and label given index for filelist files
        '''
        file = self.filelist[idx]
        #print(file)
        data, label = self.get_data(file)

        return data, label, file

    def get_data(
            self, 
            filepath: str
    ):
        '''
        given filename loads and returns file contents

        current function works for training model that predicts target boat future location given current and location of all other boats
        '''

        datapoint = np.load(filepath, allow_pickle=True)
        data_labels = datapoint.files

        #idx = int(filepath.split('/')[2])
        #a_idx = idx%3
        #agent_name = data_labels[a_idx]
        #print(type(datapoint[str(agent_name)][0]))
        data_rel = np.array(datapoint['input']).flatten()
        #last_pos = xy_pairs[-2:]  

        label_rel = np.array(datapoint['label']).flatten()
        #label_rel = label - np.tile(last_pos,4)

        #xy_pairs[0:int(len(label)/2)] = label[0:int(len(label)/2)]
        #data_rel = xy_pairs - np.tile(last_pos,22)

        return torch.from_numpy(data_rel.flatten()).to(torch.float32), torch.from_numpy(label_rel.flatten()).to(torch.float32)



