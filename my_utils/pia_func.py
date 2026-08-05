import numpy as np
import random
import time

from sklearn.linear_model import LinearRegression
from sklearn import tree, svm

from my_utils import utils
from scipy.stats import pearsonr
import numpy as np
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder



def construct_batch(args, p_num, attack_feat, aux_prop_index, aux_nonprop_index, case=1):
    nonp_num = args.sampling_size - p_num

    if p_num > len(aux_prop_index):
        p_index = np.random.choice(aux_prop_index, p_num, replace=True)
    else:
        p_index = np.random.choice(aux_prop_index, p_num, replace=False)

    if nonp_num > len(aux_nonprop_index):
        nonp_index = np.random.choice(aux_nonprop_index, nonp_num, replace=True)
    else:
        nonp_index = np.random.choice(aux_nonprop_index, nonp_num, replace=False)

    feat_matrix = np.vstack((attack_feat[p_index], attack_feat[nonp_index]))
    _gnorm = utils.calculate_norm(feat_matrix, index=[], norm_type=args.norm_type)
    _gnorm = sorted(_gnorm)

    return _gnorm


def sampling_method_target(args, attack_feat, aux_prop_index, aux_nonprop_index, prop_npy):
    start_time = time.time()
    index_list = []

    n = args.sampling_size
    r_begin = max(0, args.begin_range)
    r_end = min(100, args.end_range)
    if hasattr(args, 'interval'):
        r_end = int(1.0 / args.interval)
        r = [args.interval * random.randint(r_begin, r_end) for _ in range(args.interpolate)]
    else:
        r = [0.01 * random.randint(r_begin, r_end) for _ in range(args.interpolate)]
        
    prop_feat = []
    for frac in r:
        p_num = int(frac * n)
        _gnorm = construct_batch(args, p_num, attack_feat, aux_prop_index, aux_nonprop_index, case=1)
        prop_feat.append(_gnorm)
  
    prop_ratio = r
    if args.classifier == 'LR':
        clf = LinearRegression().fit(prop_feat, prop_ratio)
    elif args.classifier == 'DT': # 
        clf = tree.DecisionTreeRegressor().fit(prop_feat, prop_ratio)
    elif args.classifier == 'SVR':
        clf = svm.SVR().fit(prop_feat, prop_ratio)
    elif args.classifier == 'XGB':
        dtrain = xgb.DMatrix(prop_feat, label=prop_ratio)
        params = {
            'objective': 'reg:squarederror',  
            'eval_metric': 'rmse',           
            'eta': 0.1,                       
            'max_depth': 6                    
        }
        num_rounds = 100
        clf = xgb.train(params, dtrain, num_rounds)
    else:
        print(args.classifier, 'error')

    test_data = []
    pred_frac = []
    for _ in range(args.target_num):
        target_index = np.random.choice(len(prop_npy), args.sampling_size, replace=False)
        attack_feat_matrix = attack_feat[target_index]
        target_norm = utils.calculate_norm(attack_feat_matrix, index=index_list, norm_type=args.norm_type)
        target_norm = sorted(target_norm) # very important
        test_data.append(target_norm)

    if args.classifier != 'XGB':
        pred_frac = clf.predict(test_data)
    else:
        dtest = xgb.DMatrix(test_data, label=[0.0 for i in range(args.target_num)])
        pred_frac = clf.predict(dtest)

    used_time = time.time() - start_time

    return np.mean(pred_frac), used_time


def sampling_method_stack(args, features, aux_prop_index, aux_nonprop_index, prop_npy):
    start_time = time.time()
    index_list = []

    n = args.sampling_size
    r_begin = max(0, args.begin_range)
    r_end = min(100, args.end_range)
    r = [0.01 * random.randint(r_begin, r_end) for _ in range(args.interpolate)]

    prop_feat = {key: [] for key in features.keys()}
    for frac in r:
        p_num = int(frac * n)
        for key, value  in features.items():
            _gnorm = construct_batch(args, p_num, value, aux_prop_index, aux_nonprop_index, case=1)
            prop_feat[key].append(_gnorm)
    
    train_y = r
    ensemble_clf = {}
    stack_x = []
    for key in features.keys():
        train_data = prop_feat[key]
        dtrain = xgb.DMatrix(train_data, label=train_y)
        params = {
            'objective': 'reg:squarederror',  
            'eval_metric': 'rmse',           
            'eta': 0.1,                       
            'max_depth': 6                    
        }
        num_rounds = 100
        ensemble_clf[key] = xgb.train(params, dtrain, num_rounds)
        stack_x.append(ensemble_clf[key].predict(dtrain))
    
    stack_x = np.array(stack_x).T
    stack_clf = tree.DecisionTreeRegressor().fit(stack_x, train_y)

    test_data = {key:[] for key in features.keys()}
    for _ in range(args.target_num):
        target_index = np.random.choice(len(prop_npy), args.sampling_size, replace=False)
        for key in features.keys():
            target_norm = utils.calculate_norm(features[key][target_index], index=index_list, norm_type=args.norm_type)
            target_norm = sorted(target_norm) # very important
            test_data[key].append(target_norm)

    pred_frac = []
    for key in features.keys():
        dtest = xgb.DMatrix(test_data[key], label=[0.0 for _ in range(args.target_num)])
        frac = ensemble_clf[key].predict(dtest)
        pred_frac.append(frac)
    used_time = time.time() - start_time
    pred_frac = np.array(pred_frac)
    final_pred = stack_clf.predict(pred_frac.T)
    
    return np.mean(final_pred), used_time

def property_single(vfl_info, clf_name, args, attack_feat, aux_prop_index, aux_nonprop_index, prop_npy, file_name):
    args.classifier = clf_name
    pred_frac, used_time = sampling_method_target(args, attack_feat, aux_prop_index, aux_nonprop_index, prop_npy)

    row = vars(args)
    row.update(vfl_info)
    row.update({
        'pred_frac': f'{pred_frac:.4f}',
        'used_time': int(used_time)
    })
    utils.write_to_csv(row, file_name)

def property_ensemble(vfl_info, clf_name, args, features, aux_prop_index, aux_nonprop_index, prop_npy, file_name):
    args.classifier = clf_name
    pred_frac = []
    p1, used_time = sampling_method_target(args, features['b_grad'], aux_prop_index, aux_nonprop_index, prop_npy)
    pred_frac.append(p1)
    pred_frac.append(sampling_method_target(args, features['b_output'], aux_prop_index, aux_nonprop_index, prop_npy)[0])
    pred_frac.append(sampling_method_target(args, features['a_grad'], aux_prop_index, aux_nonprop_index, prop_npy)[0])
    pred_frac.append(sampling_method_target(args, features['a_output'], aux_prop_index, aux_nonprop_index, prop_npy)[0])

    args.classifier = f'en-{clf_name}'
    row = vars(args)
    row.update(vfl_info)
    row.update({
        'pred_frac': f'{sum(pred_frac)/len(pred_frac):.4f}',
        'used_time': int(used_time) * 4
    })
    utils.write_to_csv(row, file_name)

def property_stack(vfl_info, clf_name, args, features, aux_prop_index, aux_nonprop_index, prop_npy, file_name):
    pred_frac, used_time = sampling_method_stack(args, features, aux_prop_index, aux_nonprop_index, prop_npy)
    args.classifier = clf_name
    row = vars(args)
    row.update(vfl_info)
    row.update({
        'pred_frac': f'{pred_frac:.4f}',
        'used_time': int(used_time) * 4
    })
    utils.write_to_csv(row, file_name)