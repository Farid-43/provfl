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
import torch.nn.functional as F

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from dataloader import get_data
from models import model_sets
from my_utils import utils

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models import model_sets
from my_utils import utils


# baseline: attribute inference attacks
class train_NN():

    def __init__(self, trainloader, testloader, model):

        self.device = 'cuda'
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()

        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)
        self.net = self.net.to(self.device)
        
    # Training
    def train(self):

        self.net.train()
        for _, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(self.device), targets.float().to(self.device)
            logits = self.net(inputs)
            logits = torch.squeeze(logits, 1)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
    def test(self):
        self.net.eval()
        count = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                if isinstance(targets, list):
                    targets = targets[0]

                inputs, targets = inputs.to(self.device), targets.float().to(self.device)
                logits = self.net(inputs)
                logits = torch.squeeze(logits, 1)
                loss = self.criterion(logits, targets)

                pred = logits
                if count == 0:
                    y_true = targets
                    y_pred = pred
                else:
                    y_true = torch.cat((y_true, targets), axis=0)
                    y_pred = torch.cat((y_pred, pred), axis=0)
                count += 1

        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        accuracy_test = ((y_pred.round() == y_true).sum() / len(y_true))

        return accuracy_test


def baseline_NN(prop_feature, nonprop_feature, features, file='test'):
    pred_ratio = 0.0
    epoch = 50
    batch_size=32

    x_data = np.concatenate((prop_feature, nonprop_feature))
    y_data = np.concatenate((np.ones(len(prop_feature)), np.zeros(len(nonprop_feature))))
    print(x_data.shape, y_data.shape)
    f_size = x_data.shape[1]
    data = utils.tensor_data_create(x_data, y_data)

    tra_len = int(len(data)*0.9)
    val_len = len(data) - tra_len
    train_dataset, val_dataset = torch.utils.data.random_split(data, [tra_len, val_len])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # train attack model 
    model = model_sets.MLPClassifier(num_feature=f_size)
    b2 = train_NN(train_loader, test_loader, model)
    
    for i in range(epoch):
        b2.train()
        if i % 10 == 0: 
            test_acc = b2.test()
    
    # infer prop ratio
    x = features
    y = np.ones(len(x))  # random

    test_feature = utils.tensor_data_create(x, y)
    testloader = torch.utils.data.DataLoader(test_feature, batch_size=64, shuffle=False, num_workers=2)
    
    count = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            logits = model(inputs)
            pred = torch.squeeze(logits, 1)

            if count == 0:
                y_true = targets
                y_pred = pred
            else:
                y_true = torch.cat((y_true, targets), axis=0)
                y_pred = torch.cat((y_pred, pred), axis=0)
            count += 1

    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    pred_ratio = ((y_pred.round() == y_true).sum() / len(y_true)) # predicted as 1

    return pred_ratio, test_acc

def baseline_stat(args, X, y, clf_name, features):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    if clf_name == 'svm':
        clf = SVC()
    else:
        clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    pred_y = clf.predict(features)
    pred_ratio = sum(pred_y) * 1.0 / len(features)

    return acc, pred_ratio, time.time()-start_time


def baseline_pipeline(args, fname, intermediate_result, aux_prop_index, aux_nonprop_index, res_dict):
    
    print('*'*10, f"attribute inference attack + {fname}", '*'*10)
    # use simple stat model
    print(intermediate_result.shape, len(aux_prop_index), len(aux_nonprop_index))
    X = [intermediate_result[i] for i in aux_prop_index] + [intermediate_result[i] for i in aux_nonprop_index]
    y = [1] * len(aux_prop_index) + [0] * len(aux_nonprop_index)

    acc, pred_ratio, used_time = baseline_stat(args, X, y, 'svm', intermediate_result)
    res_dict.update({
        f'SM_{fname}_acc': f'{acc:.4f}',
        f'SM_{fname}_pred': f'{pred_ratio:.4f}',
        f'SM_{fname}_time': f'{used_time:.1f}',

    })

    acc, pred_ratio, used_time = baseline_stat(args, X, y, 'dt', intermediate_result)
    res_dict.update({
        f'DT_{fname}_acc': f'{acc:.4f}',
        f'DT_{fname}_pred': f'{pred_ratio:.4f}',
        f'DT_{fname}_time': f'{used_time:.1f}',

    })

    # use NN
    start_time = time.time()
    prop_feature = intermediate_result[aux_prop_index]
    nonprop_feature = intermediate_result[aux_nonprop_index]
    test_features = intermediate_result
    pred_ratio, pred_acc = baseline_NN(prop_feature, nonprop_feature, test_features)
    used_time = time.time() - start_time
    res_dict.update({
        f'NN_{fname}_acc': f'{pred_acc:.4f}',
        f'NN_{fname}_pred': f'{pred_ratio:.4f}',
        f'NN_{fname}_time': f'{used_time:.4f}',

    })

def main():
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
    
    args = parser.parse_args()
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
    

    for epoch in range(args.epochs):

        for batch_idx, (trn_X, trn_y, prop_label) in enumerate(train_loader):
            trn_X_up, trn_X_down = utils.split_data(args.dataset, trn_X) # split
            trn_X_up = trn_X_up.to(device)

            if batch_idx == 0:
                raw_a_input = trn_X_up.detach().clone()
                target_prop = prop_label
            else:
                raw_a_input = torch.cat((raw_a_input, trn_X_up.detach().clone()), axis=0)
                target_prop = torch.cat((target_prop, prop_label), axis=0)

        
        if epoch == args.attack_epoch:
            input_a_npy = raw_a_input.cpu().detach().numpy()
            prop_npy = target_prop.cpu().detach().numpy()
    

    # ============================== Baseline: classifier =======================================

    prop_index = np.where(prop_npy==1)[0]
    basline_res = {
        'gnd_frac': '%.4f' % (len(prop_index) * 1.0 / len(prop_npy)),
    } 

    baseline_pipeline(args, 'a_input', input_a_npy, aux_prop_index, aux_nonprop_index, basline_res)

    row = vars(args)
    row.update(basline_res)
    utils.write_to_csv(row, 'raw_res_%s_%s.csv' % (os.path.abspath(__file__).split('.')[0].split('_')[-1], args.dataset))



if __name__ == '__main__':
    main()
