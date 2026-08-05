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

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_data
from models import model_sets
from my_utils import utils


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
    model_name = '%s_%s_%s' % (args.dataset, args.learning_rate, args.seed)
    writer = SummaryWriter('saved_runs/%s' % (model_name))
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
        train_loader, valid_loader, prop_loader, aux_prop_index, aux_nonprop_index = get_data(args.dataset, prop_name=args.property)
        local_models.append(model_sets.MLPBottomModel(config[args.dataset]["a_dim"], config[args.dataset]["hidden_dim"])) # a: attacker
        local_models.append(model_sets.MLPBottomModel(config[args.dataset]["b_dim"], config[args.dataset]["hidden_dim"]))
        args.learning_rate = config[args.dataset]["learning_rate"]
        num_classes = config[args.dataset]["class_num"]

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

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

    criterion = criterion.to(device)

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

        for step, (trn_X, trn_y, _) in enumerate(train_loader):


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

            pred = logits
            if step == 0:
                y_true = target
                y_pred = pred
            else:
                y_true = torch.cat((y_true, target), axis=0)
                y_pred = torch.cat((y_pred, pred), axis=0)

        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        auc_train = roc_auc_score(y_true, y_pred)
        
        precision_train = precision_score(y_true, y_pred.round())
        recall_train = recall_score(y_true, y_pred.round())
        accuracy_train = ((y_pred.round() == y_true).sum() / len(y_true))
        train_loss = train_loss / len(train_loader)
        # print(f'Epoch {epoch} - Train Loss: {train_loss / len(train_loader):.4f} -  Accuracy: {accuracy_train:.4f} - Precision: {precision_train:.4f} - Recall: {recall_train:.4f} - AUC: {auc_train:.4f}')

        cur_step = (epoch + 1) * len(train_loader)

        ########### VALIDATION ###########
        loss_test, auc_test, precision_test, recall_test, accuracy_test = \
            utils.vfl_test(epoch, active_model, model_list, criterion, valid_loader, args.dataset, args.device)
        print(f'Epoch {epoch} - Valid Loss: {loss_test:.4f} -  Accuracy: {accuracy_test:.4f} - Precision: {precision_test:.4f} - Recall: {recall_test:.4f} - AUC: {auc_test:.4f}')
        
        writer.add_scalars('Loss/%s' % model_name, {'Train': train_loss, 'Test': loss_test}, epoch)
        writer.add_scalars('AUC/%s' % model_name, {'Train': auc_train, 'Test': auc_test}, epoch)
        writer.add_scalars('Precision/%s' % model_name, {'Train': precision_train, 'Test': precision_test}, epoch)
        writer.add_scalars('Recall/%s' % model_name, {'Train': recall_train, 'Test': recall_test}, epoch)
        writer.add_scalars('Accuracy/%s' % model_name, {'Train': accuracy_train, 'Test': accuracy_test}, epoch)

        # update scheduler
        for scheduler in scheduler_list:
            scheduler.step()
    
    row = vars(args)
    row.update({
        'valid_loss': f'{loss_test.item():.6f}',
        'train_loss': f'{train_loss:.6f}',
        'accuracy': f'{accuracy_test:.4f}',
        'auc': f'{auc_test:.4f}',
        'precision': f'{precision_test:.4f}',
        'recall': f'{recall_test:.4f}'
    })
    utils.write_to_csv(row, 'res_noattack_%s.csv' % (args.dataset))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("VFL")
    parser.add_argument('--dataset', type=str, default='mnist', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--k', type=int, default=3, help='num of client')
    parser.add_argument('--model', default='mlp2', help='resnet')
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--property', type=str, default='sex')
    args = parser.parse_args()

    for seed in range(10):
        args.seed = seed
        main(args)
