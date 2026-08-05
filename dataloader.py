import torch
from datasets.celeba import get_celeba_dataset
from datasets.adult import get_adult_dataset
from datasets.census import get_census_dataset
from datasets.bankmk import get_bankmk_dataset
from datasets.lawschool import get_lawschool_dataset
from datasets.health_heritage import HealthHeritage
import numpy as np
import random

from torch.utils.data import Subset


def get_data(dataset_name, prop_name, batch_size=64, val_ratio=0.2, sampling_size=2000, aligned=True, withdraw_ratio=0.0):
    '''
    prop_dataset and target_num comes from train_dataset
    adult: (44355, 111) binary classification sex/race/workclass
        Percent of positive classes: 24.50%
        Percent of target property sex=sex_Male: 66.21%
        Percent of target property race=race_White: 84.27%
        Percent of target property workclass=workclass_Private: 67.52%

    census: (299285, 511) binary classification race/sex/education
        Percent of positive classes: 6.20%
        Percent of target property sex=sex_ Female: 52.05%
        Percent of target property race=race_ Black: 10.20%
        Percent of target property education=education_ Bachelors degree(BA AB BS): 9.94%
        Percent of target property major-industry-code=major-industry-code_ Construction: 3.02%
    
    bankmk: (45211, 51) binary classification month/marital/contact
        Percent of positive classes: 11.70%
        Percent of target property month=may: 30.45%
        Percent of target property marital=married: 60.19%
        Percent of target property contact=telephone: 6.43%
            
    health_heritage #(61700, 110) binary classification  Sex/AgeAtFirstClaim
    health sex/age
        Percent of positive classes: 29.82%
        Percent of target property Sex=F: 16.13%
        Percent of target property AgeAtFirstClaim=80+: 2.68% # too few
    
    lawschool: #(96584, 39) binary classification race/resident/gender
        Percent of positive classes: 26.49%
        Percent of target property race=Black: 8.63%
        Percent of target property resident=0.0: 69.08%
        Percent of target property gender=0.0: 44.27%
    '''

    dataset_mapping = {
        'adult': get_adult_dataset,
        'census': get_census_dataset,
        'bankmk': get_bankmk_dataset,
        'lawschool': get_lawschool_dataset
    }
    
    dataset_name = dataset_name.lower()
    
    if dataset_name in dataset_mapping:
        data_func = dataset_mapping[dataset_name]
        data, prop_data = data_func(property=prop_name)
        
        for i in range(len(data)):
            data[i] = data[i] + (prop_data[i],)
    elif dataset_name.lower() == 'health':
        data, prop_data = HealthHeritage(property=prop_name).get_data()
        for i in range(len(data)):
            data[i] = data[i] + (prop_data[i],)

    elif dataset_name.lower() == 'celeba':
        num_classes, dataset = get_celeba_dataset('celeba', attr='attr', prop_name=prop_name, root='../data/')
        data_len = 200000
        data, _ = torch.utils.data.random_split(dataset, [data_len, len(dataset)-data_len])
        
    else:
        print("Dataset error!")
        return
    

    # Convert data to list and shuffle it
    data = list(data)
    random.shuffle(data)

    # Split data into training and validation sets
    val_len = int(len(data) * val_ratio)
    train_data = data[:-val_len]
    valid_data = data[-val_len:]

    # Extract property dataset
    train_prop_data = np.array([p for (_, _, p) in train_data])
    prop_index = np.where(train_prop_data == 1)[0]
    non_prop_index = np.where(train_prop_data == 0)[0]

    # withdraw defense
    if withdraw_ratio > 0.0:
        n = int(len(prop_index) * withdraw_ratio)
        prop_index = prop_index[n:]  # withdraw some property samples
        train_data = [train_data[i] for i in prop_index] + [train_data[i] for i in non_prop_index]
        train_prop_data = np.array([p for (_, _, p) in train_data])
        prop_index = np.where(train_prop_data == 1)[0]
        non_prop_index = np.where(train_prop_data == 0)[0]

    # Randomly sample indices for property and non-property data
    aux_prop_index = np.random.choice(prop_index, sampling_size, replace=(len(prop_index) < sampling_size))
    aux_nonprop_index = np.random.choice(non_prop_index, sampling_size, replace=False)

    # Modify training data if not aligned
    if not aligned:
        valid_prop_data = [p for (_, _, p) in valid_data]
        prop_data = np.array(valid_prop_data)
        prop_index = np.where(prop_data == 1)[0]
        non_prop_index = np.where(prop_data == 0)[0]

        val_aux_prop_index = np.random.choice(prop_index, sampling_size, replace=(len(prop_index) < sampling_size))
        val_aux_nonprop_index = np.random.choice(non_prop_index, sampling_size, replace=(len(non_prop_index) < sampling_size))

        for i in range(sampling_size): # val_aux_prop_index -> aux_prop_index, val_aux_nonprop_index->aux_nonprop_index
            id1 = aux_prop_index[i]
            id2 = val_aux_prop_index[i]
            train_data[id1] = valid_data[id2]

            id1 = aux_nonprop_index[i]
            id2 = val_aux_nonprop_index[i]
            train_data[id1] = valid_data[id2]

        _index = np.concatenate((val_aux_prop_index, val_aux_nonprop_index))
        property_data = [valid_data[i] for i in _index]
        prop_loader = torch.utils.data.DataLoader(property_data, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        _index = np.concatenate((aux_prop_index, aux_nonprop_index))
        property_data = [train_data[i] for i in _index]
        prop_loader = torch.utils.data.DataLoader(property_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # Create subsets and data loaders for training and validation
    train_dataset = Subset(train_data, range(len(train_data)))
    val_dataset = Subset(valid_data, range(len(valid_data)))

    print('Train dataset:%d, valid dataset:%d' % (len(train_dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # already shuffled by torch.utils.data.random_split
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    
    return train_loader, val_loader, prop_loader, aux_prop_index, aux_nonprop_index


def get_data_ratio(dataset_name, prop_name, t_value, ratio_num, batch_size=64, val_ratio=0.2, sampling_size=200):
    dataset_mapping = {
        'adult': get_adult_dataset,
        'census': get_census_dataset,
        'bankmk': get_bankmk_dataset,
        'lawschool': get_lawschool_dataset
    }
    
    dataset_name = dataset_name.lower()
    
    if dataset_name in dataset_mapping:
        data_func = dataset_mapping[dataset_name]
        data, prop_data = data_func(property=prop_name)
        
        for i in range(len(data)):
            data[i] = data[i] + (prop_data[i],)
    elif dataset_name.lower() == 'health':
        data, prop_data = HealthHeritage(property=prop_name).get_data()
        for i in range(len(data)):
            data[i] = data[i] + (prop_data[i],)

    elif dataset_name.lower() == 'celeba':
        num_classes, dataset = get_celeba_dataset('celeba', attr='attr', prop_name=prop_name, root='../data/')
        data_len = 200000
        data, _ = torch.utils.data.random_split(dataset, [data_len, len(dataset)-data_len])
        
    else:
        print("Dataset error!")
        return

    prop_data_numpy = np.array(prop_data) 
    prop_index = np.where(prop_data_numpy==1)[0]
    non_prop_index = np.setdiff1d(np.array([i for i in range(prop_data_numpy.shape[0])]), prop_index)
    print(f'All:{len(data)} Prop:{len(prop_index)} Non-prop:{len(non_prop_index)}')

    # create training and test dataset
    prop_len = int(ratio_num * t_value * 100)
    nonprop_len = int(ratio_num * (1 - t_value) * 100)
    if prop_len > len(prop_index) or nonprop_len > len(non_prop_index):
        print(f'Error - {prop_len}:{len(prop_index)} {nonprop_len}:{len(non_prop_index)}')
        return
    else:
        print(f'Assign - {prop_len}:{len(prop_index)} {nonprop_len}:{len(non_prop_index)} {dataset_name}-{prop_name}:{t_value:.2f}')


    data = list(data)
    p_index = np.random.choice(prop_index, prop_len, replace=False)
    np_index = np.random.choice(non_prop_index, nonprop_len, replace=False)
    _index = np.concatenate((p_index, np_index))
    select_data = [data[i] for i in _index]

    _len = len(select_data)
    val_len = int(_len*val_ratio)
    tra_len = _len - val_len

    train_dataset, val_dataset = torch.utils.data.random_split(select_data, [tra_len, val_len])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # create property dataset
    train_prop_data = [p for (_, _, p) in train_dataset]
    prop_data = np.array(train_prop_data)
    prop_index = np.where(prop_data==1)[0]
    non_prop_index = np.setdiff1d(np.array([i for i in range(prop_data.shape[0])]), prop_index)
    aux_prop_index = np.random.choice(prop_index, sampling_size, replace=(prop_index.shape[0] < sampling_size))
    aux_nonprop_index = np.random.choice(non_prop_index, sampling_size, replace=False)

    return train_loader, val_loader, [], aux_prop_index, aux_nonprop_index
    
if __name__ == "__main__":
    # get_data('adult', 'sex', prop_ratio=0.3)
    # get_data_prop_ratio('Adult', ['workclass', 'Private'], t_value=0.30, query_num=1000)
    # train_loader, _, _, _, _ = get_data('adult', prop_name='sex', sampling_size=30)
    # train_loader, _, _, _, _ = get_data('health', prop_name='sex', sampling_size=64)
    query = {
        'adult': ['sex', 'race', 'workclass'],
        'census': ['sex', 'race', 'education'],
        'bankmk': ['month', 'marital', 'contact'],
        'health': ['sex', 'age'],
        'lawschool': ['race', 'resident', 'gender']
    } 
    get_data('adult', prop_name='sex', val_ratio=0.3, sampling_size=2000, aligned=False, withdraw_ratio=0.1) 
    # for db, key in query.items():
    #     for p in key:
    #         get_data(db, prop_name=p, val_ratio=0.3, sampling_size=2000, aligned=False)      
