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
    


    model_list = []
    for i in range(args.k):
        if i == 0:
            if args.use_project_head == 1:
                # active_model = ClassificationModelHostTrainableHead(num_classes*2, num_classes).to(device)
                active_model = model_sets.TrainableTopModel(32, 1).to(device)
                logging.info('Trainable active party')
        else:
            model_list.append(model_sets.ClassificationModelGuest(local_models[i-1]))

    local_models = None
    model_list = [model.to(device) for model in model_list]
    criterion = nn.BCELoss()

    # weights optimizer
    optimizer_active_model = None
    if args.use_project_head == 1:
        optimizer_active_model = torch.optim.SGD(active_model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
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

    for epoch in range(args.epochs):

        cur_step = epoch * len(train_loader)
        cur_lr = optimizer_list[0].param_groups[0]['lr']

        for model in model_list:
            active_model.train()
            model.train()
        
        train_loss = 0

        for batch_idx, (trn_X, trn_y, prop_label) in enumerate(train_loader):


            trn_X_up, trn_X_down = utils.split_data(args.dataset, trn_X) # split
            
            trn_X_up = trn_X_up.to(device)
            trn_X_down = trn_X_down.to(device)
            # target = trn_y.long().to(device)
            target = trn_y.float().to(device)

            # passive party 0 generate output
            z_up = model_list[0](trn_X_up)
            z_up_clone = z_up.detach().clone()
            z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(args.device)

            # passive party 1 generate output
            z_down = model_list[1](trn_X_down)
            z_down_clone = z_down.detach().clone()
            z_down_clone = torch.autograd.Variable(z_down_clone, requires_grad=True).to(args.device)

            # active party backward
            logits = active_model(z_up_clone, z_down_clone)
            logits = torch.squeeze(logits, 1)

            loss = criterion(logits, target)
            train_loss += loss.item()

            z_gradients_up = torch.autograd.grad(loss, z_up_clone, retain_graph=True)
            z_gradients_down = torch.autograd.grad(loss, z_down_clone, retain_graph=True)

            z_gradients_up_clone = z_gradients_up[0].detach().clone()
            z_gradients_down_clone = z_gradients_down[0].detach().clone()

            # print(123, z_up_clone.shape, z_down_clone.shape, z_gradients_up_clone.shape, z_gradients_down_clone.shape)
            # 123 torch.Size([64, 16]) torch.Size([64, 16]) torch.Size([64, 16]) torch.Size([64, 16])

            if batch_idx == 0:
                a_output = z_up.detach().clone()
                a_gradient = z_gradients_up[0].detach().clone()
                b_output = z_down.detach().clone()
                b_gradient = z_gradients_down[0].detach().clone()
                target_prop = prop_label

            else:
                a_output = torch.cat((a_output, z_up.detach().clone()), axis=0)
                a_gradient = torch.cat((a_gradient, z_gradients_up[0].detach().clone()), axis=0)
                b_output = torch.cat((b_output, z_down.detach().clone()), axis=0)
                b_gradient = torch.cat((b_gradient, z_gradients_down[0].detach().clone()), axis=0)
                target_prop = torch.cat((target_prop, prop_label), axis=0)

            # update active model
            if optimizer_active_model is not None:
                optimizer_active_model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_active_model.step()

            # update passive model 0
            optimizer_list[0].zero_grad()
            weights_gradients_up = torch.autograd.grad(z_up, model_list[0].parameters(),
                                                    grad_outputs=z_gradients_up_clone)

            for w, g in zip(model_list[0].parameters(), weights_gradients_up):
                if w.requires_grad:
                    w.grad = g.detach()
            optimizer_list[0].step()

            # update passive model 1
            optimizer_list[1].zero_grad()
            weights_gradients_down = torch.autograd.grad(z_down, model_list[1].parameters(),
                                                            grad_outputs=z_gradients_down_clone)

            for w, g in zip(model_list[1].parameters(), weights_gradients_down):
                if w.requires_grad:
                    w.grad = g.detach()
            optimizer_list[1].step()
            cur_step += 1

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
        utils.vfl_test(epoch, active_model, model_list, criterion, valid_loader, args.dataset, args.device)
    print(f'Epoch {epoch} - Valid Loss: {loss_test:.4f} -  Accuracy: {accuracy_test:.4f} - Precision: {precision_test:.4f} - Recall: {recall_test:.4f} - AUC: {auc_test:.4f}')

    ############ PIA: sampling-based gradient method ###########

    vfl_info = {
        'accuracy': f'{accuracy_test:.4f}',
        'auc': f'{auc_test:.4f}',
        'precision': f'{precision_test:.4f}',
        'recall': f'{recall_test:.4f}',
        'gnd_frac': f'{len(np.where(prop_npy==1)[0]) * 1.0 / len(prop_npy):.4f}',
    }
    ############ PIA: sampling-based gradient method ###########

    features = {
        'b_grad': grad_b_npy,
        'b_output': output_b_npy,
        'a_grad': grad_a_npy,
        'a_output': output_a_npy
    }

    file_name = 'ab_%s_%s.csv' % (os.path.abspath(__file__).split('.')[0].split('_')[-1], args.dataset)

    for att_size in [100, 200, 300, 400, 500, 600, 700]:
        args.interpolate = att_size

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
        property_ensemble(vfl_info, 'XGB', args, features, aux_prop_index, aux_nonprop_index, prop_npy, file_name)


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
    parser.add_argument('--interval', type=float, default=0.01)
    parser.add_argument('--ab_test', type=str, default='test')

    args = parser.parse_args()

    for seed in range(5):
        args.seed = seed
        main(args)
