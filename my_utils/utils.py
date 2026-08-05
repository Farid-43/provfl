import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from sklearn.metrics import auc, roc_curve
import os, shutil
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import json
import pandas as pd


def tensor_data_create(features, labels):
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features])
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return dataset


def image_format_2_rgb(x):
    return x.convert("RGB")


def calculate_norm(grad, index=[], norm_type=1):
    def _selected_norm(line):
        selected_line = line if index == [] else [line[i] for i in index]
        if norm_type == 1:
            return np.sum(np.abs(selected_line)) / len(selected_line)
        elif norm_type == 2:
            return np.sum(selected_line ** 2) / len(selected_line)
        elif norm_type == 0:
            return np.sum(selected_line) / len(selected_line)
        else:
            raise ValueError("Invalid norm_type. Supported values are 1 or 2.")

    norm = []
    for i in range(len(grad)):
        row = grad[i]
        row_norm = _selected_norm(row)
        norm.append(row_norm)

    return norm

def train_val_split(labels, n_labeled_per_class, num_classes):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    # print("train_labeled_idxs:", train_labeled_idxs)
    # exit()

    return train_labeled_idxs, train_unlabeled_idxs


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def weights_init_ones(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def weights_init_normal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=1., std=1e-1)
        init.normal_(m.bias, mean=1., std=1e-1)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def keep_predict_loss(y_true, y_pred):
    # print("y_true:", y_true)
    # print("y_pred:", y_pred[0][:5])
    # print("y_true * y_pred:", (y_true * y_pred))
    return torch.sum(y_true * y_pred)

def manual_seed(seed):
    print("Setting seeds to: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#TPR@0.1%FPR TPR@1%FPR
def evaluate_mia(y_true, y_score, bounds=.01):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, threshold = roc_curve(y_true, y_score) # roc_curve(y_true, y_score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    _auc = auc(fpr, tpr)

    low1 = tpr[np.where(fpr<.001)[0][-1]]
    low2 = tpr[np.where(fpr<.01)[0][-1]]

    # low = tpr[np.where(fpr<bounds)[0][-1]]
    # print('AUC %.4f, Accuracy %.4f, TPR@%d%%FPR of %.4f'%(_auc, acc, bounds*100, low))

    return fpr, tpr, _auc, acc, low1, low2

def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])

transform_fn = transforms.Compose([
    transforms.ToTensor()
])

def get_class_i(dataset, label_set):
    gt_data = []
    gt_labels = []
    # num_cls = len(label_set)
    for j in range(len(dataset)):
        img, label = dataset[j]
        if label in label_set:
            label_new = label_set.index(label)
            gt_data.append(img if torch.is_tensor(img) else tp(img))
            gt_labels.append(label_new)
    gt_data = torch.stack(gt_data)
    return gt_data, gt_labels

def fetch_classes(num_classes):
    return np.arange(num_classes).tolist()

def fetch_data_and_label(dataset, num_classes):
    classes = fetch_classes(num_classes)
    return get_class_i(dataset, classes)

class SimpleDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i, target_i = self.data[item_idx], self.labels[item_idx]
        return torch.tensor(data_i, dtype=torch.float32), torch.tensor(target_i, dtype=torch.long)

def split_data(name, data):
    if name == 'mnist':
        data_a = data[:, :, :14, :]
        data_b = data[:, :, 14:, :]
    elif name == 'adult':
        data_a = data[:, 50:111] 
        data_b = data[:, 0:50]
    elif name == 'census':
        data_a = data[:, 255:511] 
        data_b = data[:, 0:255]
    elif name == 'bankmk':
        data_a = data[:, 25:51] 
        data_b = data[:, 0:25]
    elif name == 'lawschool':
        data_a = data[:, 20:39] 
        data_b = data[:, 0:20]
    elif name == 'health':
        data_a = data[:, 50:110] 
        data_b = data[:, 0:50]
    return data_a, data_b

def multi_party_split_data(name, data, k, dim, all_dims=0):
    res = []
    if all_dims == 0: # evenly split
        for i in range(k-1):
            res.append(data[:,i*dim:(i+1)*dim])
            # print(i, (i+1)*dim, data[:,i*dim:(i+1)*dim].shape)
        res.append(data[:, (k-1)*dim:])
    # print((k-1)*dim, -1, data[:, (k-1)*dim:].shape)
    else:
        res.append(data[:,0:dim]) # victim model
        dim1 = int((all_dims-dim)/(k-1))
        # print(0, dim, dim1)
        for i in range(k-2):
            res.append(data[:,dim+i*dim1:dim+(i+1)*dim1])
            # print(i, dim+i*dim1, dim+(i+1)*dim1)
        res.append(data[:, dim+(k-2)*dim1:])
        # print( dim+(k-2)*dim1, all_dims)
    return res

def vfl_test(epoch, active_model, model_list, criterion, valid_loader, dataset, device):
    # validate_model_list = []
    for model in model_list:
        active_model.eval()
        model.eval()

    loss_test = 0
    with torch.no_grad():
        count = 0
        for step, (val_X, val_y, _) in enumerate(valid_loader):
            val_X_a, val_X_b = split_data(dataset, val_X)
            val_X_a = val_X_a.to(device)
            val_X_b = val_X_b.to(device)
            target = val_y.view(-1).float().to(device)

            N = target.size(0)

            z_up = model_list[0](val_X_a)
            z_down = model_list[1](val_X_b)

            logits = active_model(z_up, z_down)
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

        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        auc_test = roc_auc_score(y_true, y_pred)
        
        precision_test = precision_score(y_true, y_pred.round())
        recall_test = recall_score(y_true, y_pred.round())
        accuracy_test = ((y_pred.round() == y_true).sum() / len(y_true))

        return loss / len(valid_loader), auc_test, precision_test, recall_test, accuracy_test

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def write_to_csv(row, outfile):
    df = pd.DataFrame(row,index=[0])
    if os.path.isfile(outfile):
        df.to_csv(outfile, mode='a', header=False, index=False)
    else:
        df.to_csv(outfile, mode='a', header=True, index=False)

def to_numeric(data: np.ndarray, features: dict, label: str = '', single_bit_binary: bool = False) -> np.ndarray:
    """
    Takes an array of categorical and continuous mixed type data and encodes it in numeric data. Categorical features of
    more than 2 categories are turned into a one-hot vector and continuous features are kept standing. The description
    of each feature has to be provided in the dictionary 'features'. The implementation assumes python 3.7 or higher as
    it requires the dictionary to be ordered.

    :param data: (np.ndarray) The mixed type input vector or matrix of more datapoints.
    :param features: (dict) A dictionary containing each feature of data's description. The dictionary's items have to
        be of the form: ('name of the feature', ['list', 'of', 'categories'] if the feature is categorical else None).
        The dictionary has to be ordered in the same manner as the features appear in the input array.
    :param label: (str) The name of the feature storing the label.
    :param single_bit_binary: (bool) Toggle to encode binary features in a single bit instead of a 2-component 1-hot.
    :return: (np.ndarray) The fully numeric data encoding.
    """
    num_columns = []
    n_features = 0
    for i, key in enumerate(list(features.keys())):
        if features[key] is None:
            num_columns.append(np.reshape(data[:, i], (-1, 1)))
        elif len(features[key]) == 2 and (single_bit_binary or key == label):
            num_columns.append(np.reshape(np.array([int(str(val) == str(features[key][-1])) for val in data[:, i]]), (-1, 1)))
        else:
            sub_matrix = np.zeros((data.shape[0], len(features[key])))
            col_one_place = [np.argwhere(np.array(features[key]) == str(val)) for val in data[:, i]]
            for row, one_place in zip(sub_matrix, col_one_place):
                row[one_place] = 1
            num_columns.append(sub_matrix)
        n_features += num_columns[-1].shape[-1]
    pointer = 0
    num_data = np.zeros((data.shape[0], n_features))
    for column in num_columns:
        end = pointer + column.shape[1]
        num_data[:, pointer:end] = column
        pointer += column.shape[1]
    return num_data.astype(np.float32)


def to_categorical(data: np.ndarray, features: dict, label: str = '', single_bit_binary=False, nearest_int=True) -> np.ndarray:
    """
    Takes an array of matrix of more datapoints in numerical form and turns it back into mixed type representation.
    The requirement for a successful reconstruction is that the numerical data was generated following the same feature
    ordering as provided here in the dictionary 'features'.

    :param data: (np.ndarray) The numerical data to be converted into mixed-type.
    :param features: (dict) A dictionary containing each feature of data's description. The dictionary's items have to
        be of the form: ('name of the feature', ['list', 'of', 'categories'] if the feature is categorical else None).
        The dictionary has to be ordered in the same manner as the features appear in the input array.
    :param label: (str) The name of the feature storing the label.
    :param single_bit_binary: (bool) Toggle if the binary features have been encoded in a single bit instead of a
        2-component 1-hot.
    :param nearest_int: (bool) Toggle to round to nearest integer.
    :return: (np.ndarray) The resulting mixed type data array.
    """
    cat_columns = []
    pointer = 0
    for key in list(features.keys()):
        if features[key] is None:
            if nearest_int:
                cat_columns.append(np.floor(data[:, pointer] + 0.5))
            else:
                cat_columns.append(data[:, pointer])
            pointer += 1
        elif len(features[key]) == 2 and (single_bit_binary or key == label):
            cat_columns.append([features[key][max(min(int(val + 0.5), 1), 0)] for val in data[:, pointer]])
            pointer += 1
        else:
            start = pointer
            end = pointer + len(features[key])
            hot_args = np.argmax(data[:, start:end], axis=1)
            cat_columns.append([features[key][arg] for arg in hot_args])
            pointer = end
    cat_array = None
    for cat_column in cat_columns:
        if cat_array is None:
            cat_array = np.reshape(np.array(cat_column), (data.shape[0], -1))
        else:
            cat_array = np.concatenate((cat_array, np.reshape(np.array(cat_column), (data.shape[0], -1))), axis=1)
    return cat_array


# efense
def flip_tensor(t, p):
    """
    Flip each element of the tensor t with probability p.
    """
    rand_tensor = torch.rand_like(t.float())
    mask = rand_tensor < p
    flipped_t = 1 - t
    flipped_t[mask] = t[mask]
    return flipped_t

def topk_truncate_per_row(tensor, k):
    top_values, top_indices = torch.topk(tensor, k, dim=1)
    truncated_tensor = torch.zeros_like(tensor)
    for i in range(tensor.size(0)):
        row_indices = top_indices[i]
        row_values = top_values[i]
        truncated_tensor[i][row_indices] = row_values
    return truncated_tensor