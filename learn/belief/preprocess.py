import os
import json
import shutil
import numpy as np

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def split_data(
    datadir: str,
    name: str,
    set_split: list[float] = [0.8, 0.1, 0.1],
    seed: int = 1,
):
    '''
    takes a directory for data assumed to contain numbered folders for batch with numbered individual files 
    in each folder, and creates a json file of lists containing the batches to be used in train, test, and 
    validation sets

    input
    -----
    datadir:str
        name of the folder with all training data
    name:str
        name of folder in datadir w/ specific training data
    set_split:list[float]
        list with fraction of total data for each subset [train, test, val]
    seed:int
        seed to be used 
    '''

    print('Generating data split file ...')

    folders = os.listdir(os.path.join(datadir,name))

    data_split_dict = {}
    folder_list = [int(f) for f in folders]
    #file_list = []

    #for f in folders:
    #    files = os.listdir(os.path.join(datadir,name,f))
    #    for file in files:
    #        file_list.append(os.path.join(datadir,name,f,file))

    #print(f'Total data points found: {len(file_list)}')
    train_idx = np.random.choice(folder_list, size=int(len(folder_list)*set_split[0]), replace=False)
    folder_list = [i for i in folder_list if i not in train_idx]
    test_idx = np.random.choice(folder_list, size=int(len(folder_list)*(set_split[1])/(set_split[1]+set_split[2])), replace=False)
    folder_list = [i for i in folder_list if i not in test_idx]
    val_idx = folder_list

    

    data_split_dict['test'] = test_idx.tolist()
    data_split_dict['val'] = val_idx 
    data_split_dict['train'] = train_idx.tolist()

    #make folder for data split file
    json_folder_name = os.path.join('data/datainfo',name)
    if not os.path.exists(json_folder_name):
            #logger.info("Save directory not found, creating path ...")
            mkdir(json_folder_name)

    #create and save data split file
    json_file_name = os.path.join('data/datainfo',name, 'data_split_dict_' + str(seed) + '.json')
    print(json_file_name)
    with open(json_file_name, "w") as outfile: 
        json.dump(data_split_dict, outfile,
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)

    print('Data split file created!')
        

if __name__=="__main__":
     split_data(datadir='data',name='rf_data1_32x32')
        