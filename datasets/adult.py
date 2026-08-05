# reference
# https://github.com/NazarPonochevnyi/Trained-MLP-for-Census-Income-classification/blob/main/Census_Income_notebook.ipynb
# https://github.com/itdxer/adult-dataset-analysis
import requests
import os
import pandas as pd
import numpy as np
import torch
from collections import OrderedDict
from functools import partial
from collections import defaultdict
from sklearn import preprocessing

from my_utils.utils import tensor_data_create


'''
data info: (44355, 111)
target property: race_White:84.27%, sex_Male:66.21%, workclass_Private: 67.52% # add other
positive class: 24.50%
'''

# download data
# Link: https://archive.ics.uci.edu/ml/datasets/Census+Income
def download_data():
    dataset_url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dataset_url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    filename1 = "adult_data.csv"
    filename2 = "adult_test.csv"

    if not filename1 in os.listdir():
        response = requests.get(dataset_url1)
        with open(filename1, 'wb') as file:
            file.write(response.content)
        print(filename1 + " downloaded")
    else:
        print(filename1 + " already exist")

    if not filename2 in os.listdir():
        response = requests.get(dataset_url2)
        with open(filename2, 'wb') as file:
            file.write(response.content)
        print(filename2 + " downloaded")
    else:
        print(filename2 + " already exist")


TRAIN_DATA_FILE = '../data/adult/adult.data'
TEST_DATA_FILE = '../data/adult/adult.test'

data_types = OrderedDict([
    ("age", "int"),
    ("workclass", "category"),
    ("final_weight", "int"),  # originally it was called fnlwgt
    ("education", "category"),
    ("education_num", "int"),
    ("marital_status", "category"),
    ("occupation", "category"),
    ("relationship", "category"),
    ("race", "category"),
    ("sex", "category"),
    ("capital_gain", "float"),  # required because of NaN values
    ("capital_loss", "int"),
    ("hours_per_week", "int"),
    ("native_country", "category"),
    ("income_class", "category"),
])
target_column = "income_class"

def read_dataset(path):
    return pd.read_csv(
        path,
        names=data_types,
        index_col=None,

        comment='|',  # test dataset has comment in it
        skipinitialspace=True,  # Skip spaces after delimiter
        na_values={
            'capital_gain': 99999,
            'workclass': '?',
            'native_country': '?',
            'occupation': '?',
        },
        dtype=data_types,
    )

def clean_dataset(data):
    # Test dataset has dot at the end, we remove it in order
    # to unify names between training and test datasets.
    data['income_class'] = data.income_class.str.rstrip('.').astype('category')
    
    # Remove final weight column since there is no use
    # for it during the classification.
    data = data.drop('final_weight', axis=1)
    
    # Duplicates might create biases during the analysis and
    # during prediction stage they might give over-optimistic
    # (or pessimistic) results.
    data = data.drop_duplicates()
    
    # Binarize target variable (>50K == 1 and <=50K == 0)
    data[target_column] = (data[target_column] == '>50K').astype(int)

    return data

def deduplicate(train_data, test_data):
    train_data['is_test'] = 0
    test_data['is_test'] = 1

    data = pd.concat([train_data, test_data])
    # For some reason concatenation converts this column to object
    data['native_country'] = data.native_country.astype('category')
    data = data.drop_duplicates()
    
    train_data = data[data.is_test == 0].drop('is_test', axis=1)
    test_data = data[data.is_test == 1].drop('is_test', axis=1)
    
    return train_data, test_data

def get_categorical_columns(data, cat_columns=None, fillna=True):
    if cat_columns is None:
        cat_data = data.select_dtypes('category')
    else:
        cat_data = data[cat_columns]

    if fillna:
        for colname, series in cat_data.iteritems():
            if 'Other' not in series.cat.categories:
                series = series.cat.add_categories(['Other'])

            cat_data[colname] = series.fillna('Other')
            
    return cat_data

def features_with_one_hot_encoded_categories(data, prop_key, cat_columns=None, fillna=True):
        
    cat_data = get_categorical_columns(data, cat_columns, fillna)
    one_hot_data = pd.get_dummies(cat_data)
    df = pd.concat([data, one_hot_data], axis=1)

    features = [
        'age',
        'education_num',
        'hours_per_week',
        'capital_gain',
        'capital_loss',
    ] + one_hot_data.columns.tolist()

    # normalize: task AUC 0.90->0.91
    normalize_columns = ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss']
    scaler = preprocessing.StandardScaler()
    df[normalize_columns] = scaler.fit_transform(df[normalize_columns])

    # keep target property in index 0
    columns = list(features)
    columns.remove(prop_key)  # 从列名列表中移除prop_key列
    features = [prop_key] + columns  # 将'prop_key'列添加到新的列名列表的开头
     
    X = df[features].fillna(0).values.astype(float)
    y = df[target_column].values
    p = df[prop_key].values

    return X, y, p

def get_adult_dataset(property='sex'):

    prop_list = {
        'sex': 'sex_Male',
        'race': 'race_White',
        'workclass': 'workclass_Private'
    }
    
    train_data = clean_dataset(read_dataset(TRAIN_DATA_FILE))
    test_data = clean_dataset(read_dataset(TEST_DATA_FILE))
    train_data, test_data = deduplicate(train_data, test_data)
    df = pd.concat([train_data, test_data])

    X, y, p = features_with_one_hot_encoded_categories(df, prop_list[property])
    data = tensor_data_create(X, y)

    print("Percent of positive classes: {:.2%}".format(np.mean(y)))
    print("Percent of target property {}={}: {:.2%}".format(property, prop_list[property], np.mean(p)))
    return list(data), list(p)

def get_data(prop_key='workclass_Private'):

    train_data = clean_dataset(read_dataset(TRAIN_DATA_FILE))
    test_data = clean_dataset(read_dataset(TEST_DATA_FILE))
    train_data, test_data = deduplicate(train_data, test_data)
    df = pd.concat([train_data, test_data])
    print("Percent of the positive classes in the training dataset: {:.2%}".format(np.mean(df.income_class)))

    X, y, p = features_with_one_hot_encoded_categories(df, prop_key)

    return X, y

# get_adult_dataset('sex')
# get_adult_dataset('race')
# get_adult_dataset('workclass')
# Percent of positive classes: 24.50%
# Percent of target property sex=sex_Male: 66.21%
# Percent of target property race=race_White: 84.27%
# Percent of target property workclass=workclass_Private: 67.52%


# X, y = get_data()
#  ------------------ test adult on DT classifier --------------------------------
'''

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
from functools import partial

def collect_metrics(fold_metrics, actual, predicted_proba, thresholds):
    kappa_score = partial(metrics.cohen_kappa_score, actual)
    
    fold_metrics['roc_scores'].append(metrics.roc_auc_score(actual, predicted_proba))
    fold_metrics['f1_scores'].append(metrics.f1_score(actual, predicted_proba.round()))
    fold_metrics['kappa'].append(kappa_score(predicted_proba.round()))
    fold_metrics['accuracy'].append(metrics.accuracy_score(actual, predicted_proba.round()))

    precision, recall, _ = metrics.precision_recall_curve(actual, predicted_proba)
    kappa_values = [kappa_score(predicted_proba > threshold) for threshold in thresholds]
    f1_values = [metrics.f1_score(actual,  predicted_proba > threshold) for threshold in thresholds]
    
    fold_metrics['pr_curves'].append((precision, recall))
    fold_metrics['kappa_curves'].append(kappa_values)
    fold_metrics['f1_curves'].append(f1_values)
    
def print_last_fold_stats(fold_metrics):
    print("ROC AUC score : {:.3f}".format(fold_metrics['roc_scores'][-1]))
    print("Kappa score   : {:.3f}".format(fold_metrics['kappa'][-1]))
    print("F1 score      : {:.3f}".format(fold_metrics['f1_scores'][-1]))
    print("Accuracy      : {:.3f}".format(fold_metrics['accuracy'][-1]))

def validate_model(model, X, y):
    kfold = KFold(n_splits=4, shuffle=True)
    fold_metrics= defaultdict(list)
    thresholds = np.arange(0.1, 1, 0.1)
    
    for i, (train_index, valid_index) in enumerate(kfold.split(X), start=1):
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        model.fit(x_train, y_train)

        x_predicted_probas = model.predict_proba(x_valid)
        x_predicted_proba = x_predicted_probas[:, 1]

        collect_metrics(fold_metrics, y_valid, x_predicted_proba, thresholds)
        
        print("Fold #{}".format(i))
        print_last_fold_stats(fold_metrics)
        print('-' * 30)
        
    print("")
    print("Average ROC AUC across folds  : {:.3f}".format(np.mean(fold_metrics['roc_scores'])))
    print("Average Kappa across folds    : {:.3f}".format(np.mean(fold_metrics['kappa'])))
    print("Average F1 across folds       : {:.3f}".format(np.mean(fold_metrics['f1_scores'])))
    print("Average accuracy across folds : {:.3f}".format(np.mean(fold_metrics['accuracy'])))

X, y = get_data()
model = DecisionTreeClassifier(
    min_samples_leaf=200,
    criterion='entropy',
)
validate_model(model, X, y)
'''

# (29096, 111)
# Fold #1
# ROC AUC score : 0.900
# Kappa score   : 0.553
# F1 score      : 0.650
# Accuracy      : 0.845
# ------------------------------
# Fold #2
# ROC AUC score : 0.898
# Kappa score   : 0.536
# F1 score      : 0.634
# Accuracy      : 0.841
# ------------------------------
# Fold #3
# ROC AUC score : 0.900
# Kappa score   : 0.546
# F1 score      : 0.643
# Accuracy      : 0.845
# ------------------------------
# Fold #4
# ROC AUC score : 0.902
# Kappa score   : 0.562
# F1 score      : 0.654
# Accuracy      : 0.852
   
