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

from dataloader import get_data
from models import model_sets
from my_utils import utils, defense_func
from my_utils.pia_func import *

def norm_clip(outputs, percentile=20):
    norms = torch.norm(outputs, dim=1)
    
    # threshold = torch.percentile(norms, percentile)
    k = int((percentile / 100) * norms.size(0))
    sorted_norms, _ = torch.sort(norms)
    threshold = sorted_norms[k]
    
    clipped_outputs = []
    for output in outputs:
        norm = torch.norm(output)
        if norm > threshold:
            clipped_output = output * (threshold / norm)
            clipped_outputs.append(clipped_output)
        else:
            clipped_outputs.append(output)
    
    return torch.stack(clipped_outputs)

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
        wd_raio = 0.0
        if args.defense == 'withdraw' and args.d_para > 0.0:
            wd_raio = args.d_para
        train_loader, valid_loader, prop_loader, aux_prop_index, aux_nonprop_index = get_data(args.dataset, prop_name=args.property, val_ratio=args.val_ratio, sampling_size=args.sampling_size, aligned=args.aligned, withdraw_ratio=wd_raio)
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
    _p = int(args.sampling_size*2 / args.batch_size) # label replacement

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

    # classifier for property by victim's output
    net = model_sets.DiscreteClassifier(num_feature=16).to(device)
    clf_optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


    for epoch in range(args.epochs):

        cur_step = epoch * len(train_loader)
        cur_lr = optimizer_list[0].param_groups[0]['lr']

        for model in model_list:
            active_model.train()
            model.train()
        
        train_loss = 0
        epoch_raw_norm, epoch_obs_norm, epoch_sigma = [], [], []
        print(f'###################{epoch}')
        for batch_idx, (trn_X, trn_y, prop_label) in enumerate(train_loader):


            trn_X_up, trn_X_down = utils.split_data(args.dataset, trn_X) # return data_a, data_b
            if args.defense == 'shuffle' and args.d_para > 0 and batch_idx < len(train_loader) * args.d_para:
                random.shuffle(trn_X_down)
            
            trn_X_up = trn_X_up.to(device)
            trn_X_down = trn_X_down.to(device)
            target = trn_y.float().to(device)
            prop_label = prop_label.float().to(device)

            # attacker
            z_up = model_list[0](trn_X_up)
            z_up_clone = z_up.detach().clone()
            z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(args.device)

            # victim
            z_down = model_list[1](trn_X_down)
            z_down_clone = z_down.detach().clone()
            z_down_clone = torch.autograd.Variable(z_down_clone, requires_grad=True).to(args.device)

            # ===== Apply defense to OUTPUT embeddings BEFORE top model =====
            # Routed through defense_func.apply_perturbation so this script and
            # vfl_pia_active.py cannot diverge. --defend_scope picks the party
            # (victim_only keeps a_output untouched, since the adversary's own
            # forward pass is not something a defender can reach), --defend_side
            # picks the direction.
            raw_victim_norm = defense_func.batch_mean_norm(z_down_clone, p=args.norm_type) \
                if args.log_norms else None
            out_d_para = args.d_para if args.out_para < 0 else args.out_para
            (z_up_clone, z_down_clone), dinfo = defense_func.apply_perturbation(
                [z_up_clone, z_down_clone], args, side='output', epoch=epoch)
            if args.log_norms:
                epoch_raw_norm.append(raw_victim_norm)
                epoch_obs_norm.append(defense_func.batch_mean_norm(z_down_clone, p=args.norm_type))
                epoch_sigma.append(dinfo.get('sigma', out_d_para if args.defense == 'gauss_noise' else 0.0))

            # active party backward
            logits = active_model(z_up_clone, z_down_clone)
            logits = torch.squeeze(logits, 1)

            loss = criterion(logits, target)

            train_loss += loss.item()

            z_gradients_up = torch.autograd.grad(loss, z_up_clone, retain_graph=True)
            z_gradients_down = torch.autograd.grad(loss, z_down_clone, retain_graph=True)    

            z_gradients_up_clone = z_gradients_up[0].detach().clone()
            z_gradients_down_clone = z_gradients_down[0].detach().clone()

            # ===== Apply defense to GRADIENTS =====
            # Now honours --defend_scope: under C1 the adversary computes these
            # gradients itself, so perturbing a_grad is not a deployable defense.
            # lap_noise (ProVFL's D1) and ppdl stay gradient-side by definition --
            # D1 explicitly assumes a trusted third party injects the noise.
            (z_gradients_up_clone, z_gradients_down_clone), _ = defense_func.apply_perturbation(
                [z_gradients_up_clone, z_gradients_down_clone], args, side='grad', epoch=epoch)

            # Collect intermediate results for attack evaluation.
            # Use DEFENDED outputs (z_up_clone/z_down_clone after output defense)
            # and DEFENDED gradients (z_gradients_*_clone after gradient defense).
            # This reflects what an attacker actually observes in real VFL.
            if batch_idx == 0:
                a_output = z_up_clone.detach().clone()
                a_gradient = z_gradients_up_clone.detach().clone()
                b_output = z_down_clone.detach().clone()
                b_gradient = z_gradients_down_clone.detach().clone()
                target_prop = prop_label

            else:
                a_output = torch.cat((a_output, z_up_clone.detach().clone()), axis=0)
                a_gradient = torch.cat((a_gradient, z_gradients_up_clone.detach().clone()), axis=0)
                b_output = torch.cat((b_output, z_down_clone.detach().clone()), axis=0)
                b_gradient = torch.cat((b_gradient, z_gradients_down_clone.detach().clone()), axis=0)
                target_prop = torch.cat((target_prop, prop_label), axis=0)

            # update active model
            if optimizer_active_model is not None:
                optimizer_active_model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_active_model.step()

            # update attacker model
            optimizer_list[0].zero_grad()
            weights_gradients_up = torch.autograd.grad(z_up, model_list[0].parameters(),
                                                    grad_outputs=z_gradients_up_clone)

            for w, g in zip(model_list[0].parameters(), weights_gradients_up):
                if w.requires_grad:
                    w.grad = g.detach()
            optimizer_list[0].step()

            # update victim model
            optimizer_list[1].zero_grad()
            weights_gradients_down = torch.autograd.grad(z_down, model_list[1].parameters(),
                                                            grad_outputs=z_gradients_down_clone)

            for w, g in zip(model_list[1].parameters(), weights_gradients_down):
                if w.requires_grad:
                    w.grad = g.detach()
            optimizer_list[1].step()
            cur_step += 1

        cur_step = (epoch + 1) * len(train_loader)

        # per-epoch trigger-signal trace: what the adaptive defense sees, and what it did.
        # raw_victim_norm is measured BEFORE noise, so this also works under
        # --defense None -- which is exactly the run used to pick --norm_threshold.
        if args.log_norms and epoch_raw_norm:
            utils.write_to_csv({
                'script': 'defense', 'dataset': args.dataset, 'property': args.property,
                'defense': args.defense, 'defend_scope': args.defend_scope,
                'defend_side': args.defend_side, 'adaptive_noise': args.adaptive_noise,
                'curriculum': args.curriculum, 'warmup_epochs': args.warmup_epochs,
                'norm_threshold': args.norm_threshold, 'sigma_low': args.sigma_low,
                'sigma_high': args.sigma_high, 'd_para': args.d_para, 'out_para': args.out_para,
                'seed': args.seed, 'epoch': epoch,
                'victim_norm_raw': f'{np.mean(epoch_raw_norm):.6f}',
                'victim_norm_obs': f'{np.mean(epoch_obs_norm):.6f}',
                'victim_norm_raw_med': f'{np.median(epoch_raw_norm):.6f}',
                'sigma_mean': f'{np.mean(epoch_sigma):.6f}',
                'sigma_frac_high': f'{np.mean([s >= args.sigma_high for s in epoch_sigma]):.4f}',
            }, 'norms_defense_%s.csv' % args.dataset)

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
    logging.info(f'Epoch {epoch} - Valid Loss: {loss_test:.4f} -  Accuracy: {accuracy_test:.4f} - Precision: {precision_test:.4f} - Recall: {recall_test:.4f} - AUC: {auc_test:.4f}')

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

    # write_to_csv appends headerless once the file exists, so a run that adds new
    # argparse columns must not land in a file written by an older schema. --out_csv
    # makes that explicit per campaign step.
    file_name = args.out_csv if args.out_csv else 'res_%s_%s.csv' % (
        os.path.abspath(__file__).split('.')[0].split('_')[-1], args.dataset)
    
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
    # defense
    parser.add_argument('--defense', type=str, default='None',
                        help='Defense type: shuffle|lap_noise|ppdl|withdraw|grad_clip|gauss_noise|dp_gauss|grad_sparse|random_proj') 
    parser.add_argument('--d_para', type=float, default=0.0,
                        help='Primary defense parameter for gradients (max_norm / sigma / keep_ratio / proj_dim)') 
    parser.add_argument('--d_para2', type=float, default=1.0,
                        help='Secondary defense param (noise_multiplier for dp_gauss)')
    parser.add_argument('--out_para', type=float, default=-1.0,
                        help='Output-channel defense strength; defaults to --d_para if unset (<0)')
    parser.add_argument('--defend_scope', type=str, default='victim_only',
                        choices=['victim_only', 'both'],
                        help='Which party to defend: victim_only (index 1 = b) or both (0+1). '
                             'both is undeployable under C1 -- ablation only.')
    parser.add_argument('--defend_side', type=str, default='both',
                        choices=['output', 'grad', 'both'],
                        help='Which direction to defend: output (forward embeddings, the only '
                             'channel the victim controls unilaterally), grad (needs a trusted '
                             'third party under C1), or both')
    # proposed defense: norm-triggered adaptive Gaussian noise
    parser.add_argument('--adaptive_noise', type=int, default=0,
                        help='0 = static sigma, 1 = V1 hard threshold, 2 = V2 continuous')
    parser.add_argument('--norm_threshold', type=float, default=0.0,
                        help='Trigger level for the batch mean L_p norm; read it off a '
                             '--log_norms baseline run')
    parser.add_argument('--sigma_low', type=float, default=0.005)
    parser.add_argument('--sigma_high', type=float, default=0.05)
    parser.add_argument('--sigma_alpha', type=float, default=0.05,
                        help='V2 slope: sigma = sigma_low + alpha * norm / norm_threshold')
    parser.add_argument('--curriculum', type=int, default=0,
                        help='0 = off, 1 = ramp up over --warmup_epochs (V3), 2 = decay')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    # bookkeeping -- these must match vfl_pia_active.py, which had silently drifted
    # to val_ratio 0.2 vs 0.3 and 10 vs 5 seeds, making the two result files
    # non-comparable
    parser.add_argument('--val_ratio', type=float, default=0.3,
                        help='Held-out fraction; must match vfl_pia_active.py to compare files')
    parser.add_argument('--aligned', type=int, default=1,
                        help='Use the aligned-sample split in the dataloader')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Seeds 0..n_seeds-1; must match vfl_pia_active.py for paired tests')
    parser.add_argument('--log_norms', type=int, default=0,
                        help='Write the per-epoch victim-norm / sigma trace to '
                             'norms_defense_<dataset>.csv')
    parser.add_argument('--out_csv', type=str, default='',
                        help='Explicit results file. write_to_csv appends headerless, so a run '
                             'with new columns must not land in an older file.')

    args = parser.parse_args()

    for seed in range(args.n_seeds):
        args.seed = seed
        main(args)
