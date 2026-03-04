import os
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

        with open(filelist_path, 'r') as file:
            seq_dict = json.load(file)
        
        data_list = seq_dict[self.flag]

        for data_idx in data_list:
            seq_filepath = os.path.join(data_filepath, str(data_idx))
            num_datap = int(len(os.listdir(seq_filepath)))
            suf = os.listdir(seq_filepath)[0].split('.')[-1]
            #append pairs of data and label file names ([x,y]) to total file list
            folderlist.append(seq_filepath)
            for datap in range(num_datap):
                if datap > 10:
                    filelist.append(os.path.join(seq_filepath, 'step_' + str(datap) + '.' + suf))

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
        
        label_size = 50
        magnify = 2

        datapoint = np.load(filepath, allow_pickle=True)
        data_labels = datapoint.files
        
        idx = int(filepath.split('/')[2])
        b_idx = idx%3

        #data = datapoint['target_true'] 
        new_data = []
        label = []
        boat_label = 'chaser' + str(b_idx) + '_true.npy'

        #new_data.extend(datapoint['target_true'][-21][0:5])
        new_data.extend(datapoint['target_true'][-11][0:5])
        new_data.extend(datapoint['target_true'][-1][0:5])
        new_data[5:7] = (np.array(new_data[5:7]) - new_data[0:2]) * (magnify*label_size)
        #new_data[10:12] = (np.array(new_data[10:12]) - new_data[0:2]) * (magnify*label_size)
        #new_data.extend(datapoint[boat_label][-21][0:5])
        new_data.extend(datapoint[boat_label][-11][0:5])
        new_data.extend(datapoint[boat_label][-1][0:5])
        new_data[10:12] = (np.array(new_data[10:12]) - new_data[0:2]) * (magnify*label_size)
        new_data[15:17] = (np.array(new_data[15:17]) - new_data[0:2]) * (magnify*label_size)
        #new_data[25:27] = (np.array(new_data[25:27]) - new_data[0:2]) * (magnify*label_size)

        #new_data.extend((datapoint['target_goal'] - new_data[0:2]) * (magnify*label_size))

        for i in range(len(data_labels) - 2):
            if i != b_idx:
                idx = next((j for j,label in enumerate(data_labels) if str(i) in label), -1)
                s_boat = datapoint[data_labels[idx]][-11]
                s_boat[0:2] = (s_boat[0:2] - new_data[0:2]) * (magnify*label_size)
                #label.extend(s_boat[0:5])
                label.extend(s_boat[0:2])
                label.extend([s_boat[4]])
        
        for i in range(len(data_labels) - 2):
            if i != b_idx:
                idx = next((j for j,label in enumerate(data_labels) if str(i) in label), -1)
                s_boat = datapoint[data_labels[idx]][-1]
                s_boat[0:2] = (s_boat[0:2] - new_data[0:2]) * (magnify*label_size)
                #label.extend(s_boat[0:5])
                label.extend(s_boat[0:2])
                label.extend([s_boat[4]])
        

        new_data = new_data[2:]

        #return torch.from_numpy(np.array(data).flatten()).to(torch.float32), torch.from_numpy(np.array(label).flatten()).to(torch.float32)
        return torch.from_numpy(np.array(new_data).flatten()).to(torch.float32), torch.from_numpy(np.array(label)).to(torch.float32)


    def get_data_old(
            self, 
            filepath: str
    ):
        '''
        given filename loads and returns file contents

        current function works for training model that predicts target boat future location given current and location of all other boats
        '''
        
        label_size = 50
        magnify = 2

        datapoint = np.load(filepath, allow_pickle=True)
        data_labels = datapoint.files

        #data = datapoint['target_true'] 
        new_data = []
        label = []
        label.extend(datapoint['target_true'][-1][0:5])

        new_data.extend(datapoint['target_true'][-10][0:5])
        #new_data.extend((datapoint['target_goal'] - new_data[0:2]) * (magnify*label_size))

        for i in range(len(data_labels) - 2):
            idx = next((j for j,label in enumerate(data_labels) if str(i) in label), -1)
            s_boat = datapoint[data_labels[idx]][-10]
            s_boat[0:2] = (s_boat[0:2] - new_data[0:2]) * (magnify*label_size)
            new_data.extend(s_boat[0:2])
            new_data.extend([s_boat[4]])

        label[0:2] = (np.array(label[0:2]) - new_data[0:2]) * (magnify*label_size)
        new_data = new_data[2:]

        '''
        for i,dp in enumerate(data):
            data[i][0:2] = data[i][0:2]*label_size*magnify
            new_data.append(data[i][0:5])

        #DATA IS IMAGE OF BOATS
        label = np.zeros((label_size,label_size))
        locs = []

        for i in range(len(data_labels) - 1):
            idx = next((j for j,label in enumerate(data_labels) if str(i) in label), -1)
            loc = (datapoint[data_labels[idx]][-1][0:2]*magnify + np.array([0.5,0.5]))*label_size
            locs.append(loc)
            label[np.clip(int(loc[1]),0,label_size-1),np.clip(int(loc[0]),0,label_size-1)] = 1
        
        
        avg_loc = np.clip(np.average(locs, axis=0),0,label_size-1)
        corner = (1 - np.round(avg_loc/label_size)) * (label_size-1)
        scale = np.linalg.norm( corner - avg_loc )
        for i in range(label_size):
            for j in range(label_size):
                dist = np.linalg.norm( np.array([i,j]) - avg_loc )
                label[i,j] = np.clip(1 - dist/scale, 0, 1)
        '''
        

        #return torch.from_numpy(np.array(data).flatten()).to(torch.float32), torch.from_numpy(np.array(label).flatten()).to(torch.float32)
        return torch.from_numpy(np.array(new_data).flatten()).to(torch.float32), torch.from_numpy(np.array(label)).to(torch.float32)





