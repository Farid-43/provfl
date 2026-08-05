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

from dataloader import get_data
from models import model_sets
from my_utils import utils
from my_utils.pia_func import *

def multi_party_test(all_dims, dim, top_model, model_list, criterion, valid_loader, dataset, device):
    
    top_model.eval()
    for model in model_list:
        model.eval()
    k = len(model_list)

    loss_test = 0
    with torch.no_grad():
        count = 0
        for step, (val_X, val_y, _) in enumerate(valid_loader):

            target = val_y.float().to(device)
            party_input_list = utils.multi_party_split_data(dataset, val_X, k, dim, all_dims) # split
            # the first one is victim, and the last one is attack
            z_up_list = []
            z_up_clone_list = []
            for i in range(k):
                party_input_list[i] = party_input_list[i].to(device)
                z_up = model_list[i](party_input_list[i])
                z_up_clone = z_up.detach().clone()
                z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(device)
                z_up_list.append(z_up)
                z_up_clone_list.append(z_up_clone)

            logits = top_model(z_up_clone_list)
            logits = torch.squeeze(logits, 1)

            loss = criterion(logits, target)
            loss_test += loss.item()

            pred = logits
            if count == 0:
                y_true = target
                y_pred = pred
            else:
                y_true = torch.cat((y_true, target), axis=0)
                y_pred = torch.cat((y_pred, pred), axis=0)
            count += 1

            if step > len(valid_loader) / 10:
                continue # use 10% test
             

        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        auc_test = roc_auc_score(y_true, y_pred)
        precision_test = precision_score(y_true, y_pred.round())
        recall_test = recall_score(y_true, y_pred.round())
        accuracy_test = ((y_pred.round() == y_true).sum() / len(y_true))

        return loss / len(valid_loader), auc_test, precision_test, recall_test, accuracy_test

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
        passive_dim = int(config[args.dataset]['input_dim'] * args.victim_feature)
        other_dim = int((config[args.dataset]['input_dim'] - passive_dim)/2)
        active_dim = config[args.dataset]['input_dim'] - passive_dim - other_dim*(args.k-2)
        # print('all dim:', config[args.dataset]['input_dim'])
        # print('victim dim:', passive_dim)
        # print('other dim:', other_dim)
        # print('active dim:', active_dim)

        # passive party
        local_models.append(model_sets.MLPBottomModel(passive_dim, config[args.dataset]["hidden_dim"])) # victim model
        for i in range(args.k-2):
            local_models.append(model_sets.MLPBottomModel(other_dim, config[args.dataset]["hidden_dim"])) # other model
        # active party
        local_models.append(model_sets.MLPBottomModel(active_dim, config[args.dataset]["hidden_dim"]))
        
        args.learning_rate = config[args.dataset]["learning_rate"]
        num_classes = config[args.dataset]["class_num"]
    
    
    print('bottom models:', len(local_models))
    
    model_list = []
    if args.use_project_head == 1:
        top_model = model_sets.TrainableTopModel_Multi_Party(16*args.k, 1).to(device)
        logging.info('Trainable active party')

    for i in range(args.k):
        model_list.append(model_sets.ClassificationModelGuest(local_models[i]))

    local_models = None
    model_list = [model.to(device) for model in model_list]
    criterion = nn.BCELoss()

    # weights optimizer
    optimizer_active_model = None
    if args.use_project_head == 1:
        optimizer_active_model = torch.optim.SGD(top_model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            for model in model_list]
    else:
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)
            for model in model_list]
    
    scheduler_list = []
    if args.learning_rate == 0.025:
        if optimizer_active_model is not None:
            scheduler_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_active_model, float(args.epochs)))
        scheduler_list = scheduler_list + [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
            for optimizer in optimizer_list]
    else:
        if optimizer_active_model is not None:
            scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_active_model, args.decay_period, gamma=args.gamma))
        scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                          in optimizer_list]


    all_dims = config[args.dataset]['input_dim']
    for epoch in range(args.epochs):

        cur_step = epoch * len(train_loader)
        cur_lr = optimizer_list[0].param_groups[0]['lr']

        top_model.train()
        for model in model_list:
            model.train()
        
        train_loss = 0

        for batch_idx, (trn_X, trn_y, prop_label) in enumerate(train_loader):

            target = trn_y.float().to(device)
            party_input_list = utils.multi_party_split_data(args.dataset, trn_X, args.k, passive_dim, all_dims) # split
            # the first one is victim, and the last one is attack
            z_up_list = []
            z_up_clone_list = []
            for i in range(args.k):
                # print(i, party_input_list[i].shape)
                party_input_list[i] = party_input_list[i].to(device)
                z_up = model_list[i](party_input_list[i])
                z_up_clone = z_up.detach().clone()
                z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(args.device)
                z_up_list.append(z_up)
                z_up_clone_list.append(z_up_clone)

            logits = top_model(z_up_clone_list)
            logits = torch.squeeze(logits, 1)

            loss = criterion(logits, target)
            train_loss += loss.item()

            z_gradients_list = []
            z_gradients_clone_list = []
            for i in range(args.k):
                z_gradients = torch.autograd.grad(loss, z_up_clone_list[i], retain_graph=True)
                z_gradients_clone = z_gradients[0].detach().clone()
                z_gradients_list.append(z_gradients)
                z_gradients_clone_list.append(z_gradients_clone)

            if batch_idx == 0:
                a_output = z_up_list[-1].detach().clone()
                a_gradient = z_gradients_list[-1][0].detach().clone()
                b_output = z_up_list[0].detach().clone()
                b_gradient = z_gradients_list[0][0].detach().clone()
                target_prop = prop_label

            else:
                a_output = torch.cat((a_output, z_up_list[-1].detach().clone()), axis=0)
                a_gradient = torch.cat((a_gradient, z_gradients_list[-1][0].detach().clone()), axis=0)
                b_output = torch.cat((b_output, z_up_list[0].clone()), axis=0)
                b_gradient = torch.cat((b_gradient, z_gradients_list[0][0].detach().clone()), axis=0)
                target_prop = torch.cat((target_prop, prop_label), axis=0)

            # update top model
            if optimizer_active_model is not None:
                optimizer_active_model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_active_model.step()

            # update bottom
            for i in range(args.k):
                optimizer_list[i].zero_grad()
                weights_gradients_up = torch.autograd.grad(z_up_list[i], model_list[i].parameters(),
                                                        grad_outputs=z_gradients_clone_list[i])

                for w, g in zip(model_list[i].parameters(), weights_gradients_up):
                    if w.requires_grad:
                        w.grad = g.detach()
                optimizer_list[i].step()

            cur_step += 1
            if batch_idx > len(train_loader) / 10:
                continue # use 10% training

        cur_step = (epoch + 1) * len(train_loader)
        
        # update scheduler
        for scheduler in scheduler_list:
            scheduler.step()
    
        # store intermediate outputs
        if args.save_feat:
            npy_dir = 'feat_%s_%s' % (args.dataset, args.property)
            if not os.path.exists(npy_dir):
                os.makedirs(npy_dir)
            np.savez('%s/epoch_%d.npz' % (npy_dir, epoch), \
                    a_output.cpu().detach().numpy(), \
                    a_gradient.cpu().detach().numpy(), \
                    b_output.cpu().detach().numpy(), \
                    b_gradient.cpu().detach().numpy(), \
                    target_prop.cpu().detach().numpy())
        
        if epoch == args.attack_epoch:
            output_a_npy = a_output.cpu().detach().numpy()
            grad_a_npy = a_gradient.cpu().detach().numpy()
            output_b_npy = b_output.cpu().detach().numpy()
            grad_b_npy = b_gradient.cpu().detach().numpy()
            prop_npy = target_prop.cpu().detach().numpy()
    
    ############ FINAL VALIDATION ###########
    loss_test, auc_test, precision_test, recall_test, accuracy_test = \
        multi_party_test(all_dims, passive_dim, top_model, model_list, criterion, valid_loader, args.dataset, args.device)
    print(f'Epoch {epoch} - Valid Loss: {loss_test:.4f} -  Accuracy: {accuracy_test:.4f} - Precision: {precision_test:.4f} - Recall: {recall_test:.4f} - AUC: {auc_test:.4f}')

    ############ PIA: sampling-based gradient method ###########
    vfl_info = {
        'accuracy': f'{accuracy_test:.4f}',
        'auc': f'{auc_test:.4f}',
        'precision': f'{precision_test:.4f}',
        'recall': f'{recall_test:.4f}',
        'gnd_frac': f'{len(np.where(prop_npy==1)[0]) * 1.0 / len(prop_npy):.4f}',
    }

    features = {
        'b_grad': grad_b_npy,
        'b_output': output_b_npy,
        'a_grad': grad_a_npy,
        'a_output': output_a_npy
    }

    file_name = 'ab_%s_%s.csv' % (os.path.abspath(__file__).split('.')[0].split('_')[-1], args.dataset)
    

    clf = 'DT'
    args.attack_feat = 'b_grad'
    property_single(vfl_info, clf, args, features['b_grad'], aux_prop_index, aux_nonprop_index, prop_npy, file_name)
    args.attack_feat = 'b_output'
    property_single(vfl_info, clf, args, features['b_output'], aux_prop_index, aux_nonprop_index, prop_npy, file_name)
    args.attack_feat = 'a_grad'
    property_single(vfl_info, clf, args, features['a_grad'], aux_prop_index, aux_nonprop_index, prop_npy, file_name)
    args.attack_feat =  'a_output'
    property_single(vfl_info, clf, args, features['a_output'], aux_prop_index, aux_nonprop_index, prop_npy, file_name)

    clf = 'XGB'
    args.attack_feat = 'b_grad'
    property_single(vfl_info, clf, args, features['b_grad'], aux_prop_index, aux_nonprop_index, prop_npy, file_name)
    args.attack_feat = 'b_output'
    property_single(vfl_info, clf, args, features['b_output'], aux_prop_index, aux_nonprop_index, prop_npy, file_name)
    args.attack_feat = 'a_grad'
    property_single(vfl_info, clf, args, features['a_grad'], aux_prop_index, aux_nonprop_index, prop_npy, file_name)
    args.attack_feat =  'a_output'
    property_single(vfl_info, clf, args, features['a_output'], aux_prop_index, aux_nonprop_index, prop_npy, file_name)

    args.attack_feat = 'all'
    property_ensemble(vfl_info, 'DT', args, features, aux_prop_index, aux_nonprop_index, prop_npy,file_name)
    property_ensemble(vfl_info, 'XGB', args, features, aux_prop_index, aux_nonprop_index, prop_npy, file_name)
    property_stack(vfl_info, 'stack', args, features, aux_prop_index, aux_nonprop_index, prop_npy, file_name)





if __name__ == '__main__':
    parser = argparse.ArgumentParser("VFL")
    # model
    parser.add_argument('--dataset', type=str, default='census')
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
    # test multi-party
    parser.add_argument('--victim_feature', type=float, default=0.5) # uneven feature split
    args = parser.parse_args()

    for seed in range(5):
        args.seed = seed
        main(args)
