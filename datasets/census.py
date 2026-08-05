import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from my_utils.utils import tensor_data_create


def get_adult_columns():

    column_names = [
        "age",
        "workclass",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]

    cont_columns = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    cat_columns = sorted(list(set(column_names).difference(cont_columns)))

    return cat_columns, cont_columns


def get_census_columns():
    """Returns names of categorical and continuous columns for census dataset"""

    column_names = [
        "age",
        "class-of-worker",
        "detailed-industry-recode",
        "detailed-occupation-recode",
        "education",
        "wage-per-hour",
        "enroll-in-edu-inst-last-wk",
        "marital-stat",
        "major-industry-code",
        "major-occupation-code",
        "race",
        "hispanic-origin",
        "sex",
        "member-of-a-labor-union",
        "reason-for-unemployment",
        "full-or-part-time-employment-stat",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "tax-filer-stat",
        "region-of-previous-residence",
        "state-of-previous-residence",
        "detailed-household-and-family-stat",
        "detailed-household-summary-in-household",
        "instance-weight",
        "migration-code-change-in-msa",
        "migration-code-change-in-reg",
        "migration-code-move-within-reg",
        "live-in-this-house-1-year-ago",
        "migration-prev-res-in-sunbelt",
        "num-persons-worked-for-employer",
        "family-members-under-18",
        "country-of-birth-father",
        "country-of-birth-mother",
        "country-of-birth-self",
        "citizenship",
        "own-business-or-self-employed",
        "fill-inc-questionnaire-for-veterans-admin",
        "veterans-benefits",
        "weeks-worked-in-year",
        "year",
    ]

    cont_columns = [
        "age",
        "wage-per-hour",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "instance-weight",
        "num-persons-worked-for-employer",
        "weeks-worked-in-year",
    ]
    cat_columns = sorted(list(set(column_names).difference(cont_columns)))

    return cat_columns, cont_columns


def generate_class_imbalance(data, target_class, target_ratio):
    Nt = sum(data["class"] == target_class)
    No = (1 - target_ratio) * Nt / target_ratio

    tgt_idx = data["class"] == target_class
    tgt_data = data[tgt_idx]
    other_data = data[~tgt_idx]
    other_data, _ = train_test_split(
        other_data, train_size=No / other_data.shape[0], random_state=21
    )

    data = pd.concat([tgt_data, other_data])
    data = data.sample(frac=1).reset_index(drop=True)

    return data


def generate_subpopulation(
    df, categories=[], target_attributes=[], return_not_subpop=False
):
    """Given a list of categories and target attributes, generate a dataframe with only those targets
    ...
    Parameters
    ----------
        df : Pandas Dataframe
            A pandas dataframe
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_not_subpop : bool
            If True, also return df/subpopulation

    ...
    Returns
    -------
        subpop : Pandas Dataframe
            The dataframe containing the target subpopulation
        not_subpop : Pandas Dataframe (optional)
            df/subpopulation
    """

    indices_with_each_target_prop = []

    for category, target in zip(categories, target_attributes):

        indices_with_each_target_prop.append(df[category] == target)

    indices_with_all_targets = np.array(
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )

    if return_not_subpop:
        return df[indices_with_all_targets].copy(), df[~indices_with_all_targets].copy()
    else:
        return df[indices_with_all_targets].copy()


def all_dfs_to_one_hot(dataframes, cat_columns=[], class_label=None):
    """Transform multiple dataframes to one-hot concurrently so that
    they maintain consistency in their columns

        ...
        Parameters
        ----------
            dataframes : list
                A list of pandas dataframes to convert to one-hot
            cat_columns : list
                A list containing all the categorical column names for the list of
                dataframes
            class_label : str
                The column label for the training label column

        ...
        Returns
        -------
            dataframes_OH : list
                A list of one-hot encoded dataframes. The output is ordered in the
                same way the list of dataframes was input
    """

    keys = list(range(len(dataframes)))

    # Make copies of dataframes to not accidentally modify them
    dataframes = [df.copy() for df in dataframes]
    cont_columns = sorted(
        list(set(dataframes[0].columns).difference(cat_columns + [class_label]))
    )

    # Get dummy variables over union of all columns.
    # Keys keep track of individual dataframes to
    # split later
    temp = pd.get_dummies(pd.concat(dataframes, keys=keys), columns=cat_columns)

    # Normalize continuous values
    temp[cont_columns] = temp[cont_columns] / temp[cont_columns].max()

    if class_label:
        temp["label"] = temp[class_label]
        temp = temp.drop([class_label], axis=1)

    # Return the dataframes as one-hot encoded in the same order
    # they were given
    return [temp.xs(i) for i in keys]


# --------------------------------------------------------------------------------------

def load_census_data(
    one_hot=True, custom_balance=False, target_class=1, target_ratio=0.1
):
    """Load the data from the census income (KDD) dataset

    ...
    Parameters
    ----------
        one_hot : bool
            Indicates whether one-hot versions of the data should be loaded.
            The one-hot dataframes also have normalized continuous values

    Returns
    -------
        dataframes : tuple
            A tuple of dataframes that contain the census income dataset.
            They are in the following order [train, test, one-hot train, one-hot test]
    """

    filename_train = "../data/census/census-income.data"
    filename_test = "../data/census/census-income.test"

    column_names = [
        "age",
        "class-of-worker",
        "detailed-industry-recode",
        "detailed-occupation-recode",
        "education",
        "wage-per-hour",
        "enroll-in-edu-inst-last-wk",
        "marital-stat",
        "major-industry-code",
        "major-occupation-code",
        "race",
        "hispanic-origin",
        "sex",
        "member-of-a-labor-union",
        "reason-for-unemployment",
        "full-or-part-time-employment-stat",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "tax-filer-stat",
        "region-of-previous-residence",
        "state-of-previous-residence",
        "detailed-household-and-family-stat",
        "detailed-household-summary-in-household",
        "instance-weight",
        "migration-code-change-in-msa",
        "migration-code-change-in-reg",
        "migration-code-move-within-reg",
        "live-in-this-house-1-year-ago",
        "migration-prev-res-in-sunbelt",
        "num-persons-worked-for-employer",
        "family-members-under-18",
        "country-of-birth-father",
        "country-of-birth-mother",
        "country-of-birth-self",
        "citizenship",
        "own-business-or-self-employed",
        "fill-inc-questionnaire-for-veterans-admin",
        "veterans-benefits",
        "weeks-worked-in-year",
        "year",
    ]

    uncleaned_df_train = pd.read_csv(filename_train, header=None)
    uncleaned_df_test = pd.read_csv(filename_test, header=None)

    mapping = {i: column_names[i] for i in range(len(column_names))}
    mapping[len(column_names)] = "class"
    
    uncleaned_df_train = uncleaned_df_train.rename(columns=mapping)
    uncleaned_df_test = uncleaned_df_test.rename(columns=mapping)
    # uncleaned_df_train = uncleaned_df_train[new_order]

    cont_columns = [
        "age",
        "wage-per-hour",
        "capital-gains",
        "capital-losses",
        "dividends-from-stocks",
        "instance-weight",
        "num-persons-worked-for-employer",
        "weeks-worked-in-year",
    ]
    cat_columns = sorted(list(set(column_names).difference(cont_columns)))


    encoder = LabelEncoder()
    uncleaned_df_train["class"] = encoder.fit_transform(uncleaned_df_train["class"])

    encoder = LabelEncoder()
    uncleaned_df_test["class"] = encoder.fit_transform(uncleaned_df_test["class"])

    uncleaned_df_train = uncleaned_df_train.drop(
        uncleaned_df_train[uncleaned_df_train["class"] == 2].index
    )
    uncleaned_df_test = uncleaned_df_test.drop(
        uncleaned_df_test[uncleaned_df_test["class"] == 2].index
    )

    if custom_balance == True:
        uncleaned_df_train = generate_class_imbalance(
            data=uncleaned_df_train,
            target_class=target_class,
            target_ratio=target_ratio,
        )
        uncleaned_df_test = generate_class_imbalance(
            data=uncleaned_df_test, target_class=target_class, target_ratio=target_ratio
        )

    if one_hot:
        # Normalize continous values
        uncleaned_df_train[cont_columns] = (
            uncleaned_df_train[cont_columns] / uncleaned_df_train[cont_columns].max()
        )
        uncleaned_df_test[cont_columns] = (
            uncleaned_df_test[cont_columns] / uncleaned_df_test[cont_columns].max()
        )

        uncleaned_df = pd.concat([uncleaned_df_train, uncleaned_df_test])

        dummy_tables = [
            pd.get_dummies(uncleaned_df[column], prefix=column)
            for column in cat_columns
        ]
        
        dummy_tables.append(uncleaned_df.drop(labels=cat_columns, axis=1))
        one_hot_df = pd.concat(dummy_tables, axis=1)

        one_hot_df["labels"] = one_hot_df["class"]
        one_hot_df = one_hot_df.drop(["class"], axis=1)

        one_hot_df_train = one_hot_df[: len(uncleaned_df_train)]
        one_hot_df_test = one_hot_df[len(uncleaned_df_train) :]

        return  one_hot_df_train, one_hot_df_test

    return uncleaned_df_train, uncleaned_df_test



def get_census_dataset(property): # [name, value]
    prop_list = {
        'race': 'race_ Black',
        'sex': 'sex_ Female',
        'education': 'education_ Bachelors degree(BA AB BS)',
        'major-industry-code': 'major-industry-code_ Construction'
    }

    df_train, df_test = load_census_data(one_hot=True)
    df = pd.concat([df_train, df_test], axis=0)

    # keep target property in index 0
    prop_key = prop_list[property]
    columns = list(df.columns)
    columns.remove(prop_key)  # 从列名列表中移除prop_key列
    new_order = [prop_key] + columns  # 将'prop_key'列添加到新的列名列表的开头
    df = df[new_order]
    p = df[prop_key].values

    df = df.reset_index(drop=True)
    y_data = df['labels'].to_numpy()
    df = df.drop(['labels'], axis=1)
    x_data = df.to_numpy()

    length = len(x_data) # only include 10k records
    indices = np.random.choice(length, 100000, replace=False)

    x_data = x_data[indices]
    y_data = y_data[indices]
    p = p[indices]

    print("Percent of positive classes: {:.2%}".format(np.mean(y_data)))
    print("Percent of target property {}={}: {:.2%}".format(property, prop_list[property], np.mean(p)))
    data = tensor_data_create(x_data, y_data)

    return list(data), list(p)

# get_census_dataset('race')
# get_census_dataset('sex')
# get_census_dataset('education')
# get_census_dataset('major-industry-code')

# Percent of positive classes: 6.20%
# Percent of target property race=race_ Black: 10.20%
# Percent of target property sex=sex_ Female: 52.05%
# Percent of target property education=education_ Bachelors degree(BA AB BS): 9.94%
# Percent of target property major-industry-code=major-industry-code_ Construction: 3.02%


def get_data():

    df_train, df_test = load_census_data(one_hot=True)
    df = pd.concat([df_train, df_test], axis=0)

    df = df.reset_index(drop=True)
    y_data = df['labels'].to_numpy()
    df = df.drop(['labels'], axis=1)
    x_data = df.to_numpy()

    return x_data, y_data

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


# Fold #1
# ROC AUC score : 0.933
# Kappa score   : 0.457
# F1 score      : 0.481
# Accuracy      : 0.950
# ------------------------------
# Fold #2
# ROC AUC score : 0.931
# Kappa score   : 0.439
# F1 score      : 0.464
# Accuracy      : 0.949
# ------------------------------
# Fold #3
# ROC AUC score : 0.932
# Kappa score   : 0.452
# F1 score      : 0.475
# Accuracy      : 0.950
# ------------------------------
# Fold #4
# ROC AUC score : 0.932
# Kappa score   : 0.452
# F1 score      : 0.475
# Accuracy      : 0.951
# ------------------------------

# Average ROC AUC across folds  : 0.932
# Average Kappa across folds    : 0.450
# Average F1 across folds       : 0.474
# Average accuracy across folds : 0.950