import os, uuid, pickle,csv

import numpy as np

import torch
from torch.utils.data import Dataset

from src.config import DATASET_PATHS


def list_dir(dir):
    return os.listdir(dir)

def make_dir(new_directory_path):
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)
        return True
    else:
        return False

def path_exists(check_path):
    return os.path.exists(check_path)

def delete_path(path2delete):
    os.remove(path2delete)

def pickle_dump(data,path):

    dir='/'.join(path.split('/')[:-1])
    make_dir(dir)

    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


# utils for Data management
def load_wrapper(path2data):
    
    extension = path2data.split('.')[-1]
    
    if extension=='pt':
        return torch.load(path2data)
    elif extension=='pkl':
        return pickle_load(path2data)
    elif extension=='npy':
        return np.load(path2data)

class RadarDataset(Dataset):
    def __init__(self,input_tensors,target_tensors):
        assert input_tensors.shape[0]==target_tensors.shape[0] 

        self.input_tensors = input_tensors
        self.target_tensors = target_tensors
        self.nof_samples=self.input_tensors.shape[0]

    def __getitem__(self, idx):
        # NCxyz
        # NCktr
        return self.input_tensors[idx,:,:,:,:],self.target_tensors[idx,:,:,:,:]

    def __len__(self):
        return self.nof_samples

def load_dataset(data_name, identifier):
    path2data=DATASET_PATHS[data_name][identifier]
    return load_wrapper(path2data)



def get_uuid():
    return uuid.uuid4().hex


def update_csv(path,new_row):
    with open(path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_row)


import json
def update_json(path,data):
    if not path_exists(path):
        foo = []
        foo.append(data)
        with open(path, mode='w') as f:
            f.write(json.dumps(foo, indent=4))
    else:
        with open(path) as feedsjson:
            feeds = json.load(feedsjson)
        feeds.append(data)
        with open(path, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))

def load_json(path):
    with open(path, 'r') as info_file:
        data=info_file.read()
        data=json.loads(data)
    return data

'''def dump_with_id(identifier, data):
    dump_dict={}
    dump_dict['identifier']
'''