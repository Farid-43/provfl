import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch
import pandas as pd

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from scipy.stats import pearsonr


from dataloader import get_data
from models import model_sets
from my_utils import utils
from my_utils.pia_func import sampling_method_target



def main(args):

    config = utils.load_config('config.json')
    args.learning_rate = config[args.dataset]["learning_rate"]
    args.epochs = config[args.dataset]["epochs"] if args.epochs == 0 else args.epochs

    name = 'experiment_result_{}/{}-{}-{}-{}-{}-{}'.format(
        args.dataset, args.model, args.batch_size, args.seed, args.use_project_head, \
            args.learning_rate, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(name)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = device

    logging.info('***** USED DEVICE: {}'.format(device))

    # set seed for target label and poisoned and target sample selection
    manual_seed = 42
    random.seed(manual_seed)

    # set seed for model initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


    ##### read dataset and set model
    local_models = []
    if args.dataset == 'mnist':
        half_dim = 14
        num_classes = 10
        train_dst = datasets.MNIST("./dataset", download=True, train=True, transform=utils.transform_fn)
        data, label = utils.fetch_data_and_label(train_dst, num_classes)
        train_dst = utils.SimpleDataset(data, label)
        test_dst = datasets.MNIST("./dataset", download=True, train=False, transform=utils.transform_fn)
        data, label = utils.fetch_data_and_label(test_dst, num_classes)
        test_dst = utils.SimpleDataset(data, label)
        train_loader = DataLoader(train_dst, batch_size=128)
        valid_loader = DataLoader(test_dst, batch_size=128)
        #
        for i in range(args.k-1):
            backbone = model_sets.MLPBottomModel(half_dim * half_dim * 2, num_classes)
        local_models.append(backbone)

    elif args.dataset in ['adult', 'census', 'bankmk', 'lawschool', 'health']:
        train_loader, valid_loader, prop_loader, aux_prop_index, aux_nonprop_index = get_data(args.dataset, prop_name=args.property, sampling_size=args.sampling_size)
        local_models.append(model_sets.MLPBottomModel(config[args.dataset]["a_dim"], config[args.dataset]["hidden_dim"])) # a: attacker
        local_models.append(model_sets.MLPBottomModel(config[args.dataset]["b_dim"], config[args.dataset]["hidden_dim"]))
        args.learning_rate = config[args.dataset]["learning_rate"]
        num_classes = config[args.dataset]["class_num"]
    



    for batch_idx, (trn_X, trn_y, prop_label) in enumerate(train_loader):

        trn_X_up, trn_X_down = utils.split_data(args.dataset, trn_X) # split
        trn_X_up = trn_X_up.to(device)
        trn_X_down = trn_X_down.to(device)
        target = trn_y.float().to(device)

        if batch_idx == 0:
            a_feat = trn_X_up.detach().clone()
            task_label = target.detach().clone()
            target_prop = prop_label

        else:
            a_feat = torch.cat((a_feat, trn_X_up.detach().clone()), axis=0)
            task_label = torch.cat((task_label, target.detach().clone()), axis=0)
            target_prop = torch.cat((target_prop, prop_label), axis=0)

    feat_a_npy = a_feat.cpu().detach().numpy()
    task_label_npy = task_label.cpu().detach().numpy()
    prop_npy = target_prop.cpu().detach().numpy()


    num_columns = feat_a_npy.shape[1]
    a_feat_cc = []
    for i in range(num_columns):
        column = feat_a_npy[:, i]  
        cc, p_value = pearsonr(column, prop_npy)
        a_feat_cc.append(cc)

    a_feat_cc = np.array(a_feat_cc)
    y_task_cc, _ = pearsonr(task_label_npy, prop_npy)

    row = {
        'dataset': args.dataset,
        'property': args.property,
        'max_a_feat_cc': f'{np.max(a_feat_cc):.4f}',
        'ave_a_feat_cc': f'{np.mean(a_feat_cc):.4f}',
        'mid_a_feat_cc': f'{np.median(a_feat_cc):.4f}',
        'y_task_cc':f'{y_task_cc:.4f}'
    }
    utils.write_to_csv(row, 'ab_feat_correlation.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("VFL")
    # model
    parser.add_argument('--dataset', type=str, default='mnist', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=0, help='num of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--k', type=int, default=3, help='num of client')
    parser.add_argument('--model', default='mlp2', help='resnet')
    parser.add_argument('--use_project_head', type=int, default=1)
    # attack
    parser.add_argument('--norm_type', type=int, default=1, help='use norm type to calculate feature distance')
    parser.add_argument('--save_feat', type=int, default=0, help='save intermediate outputs')
    parser.add_argument('--sampling_size', type=int, default=2000, help='num of overlapping samples')
    parser.add_argument('--select_size', type=int, default=0, help='select correlated neurons') 
    parser.add_argument('--attack_epoch', type=int, default=18, help='epoch used in attack')
    parser.add_argument('--interpolate', type=int, default=200)
    parser.add_argument('--classifier', type=str, default='DT') # LR/simi
    parser.add_argument('--property', type=str, default='sex')
    parser.add_argument('--attack_feat', type=str, default='b_grad', help='attack model feed into') # b_grad, b_output, a_grad, a_output
    parser.add_argument('--begin_range', type=int, default=0)
    parser.add_argument('--end_range', type=int, default=100)
    parser.add_argument('--target_num', type=int, default=100)

    
    args = parser.parse_args()

    query = {
        'adult': ['sex', 'race', 'workclass'],
        'census': ['sex', 'race', 'education'],
        'bankmk': ['month', 'marital', 'contact'],
        'health': ['sex', 'age'],
        'lawschool': ['race', 'resident', 'gender']
    }

    for db, values in query.items():
        for prop in values:
            args.dataset = db
            args.property = prop
            main(args)
