import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from my_utils.utils import tensor_data_create


def get_bankmk_columns():

    column_names = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y"
    ]

    cont_columns = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous"
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


def load_bankmk_data(one_hot=True, custom_balance=False, target_class=1, target_ratio=0.3, prop=None):
    """Load the bankmk dataset."""

    prop_name, prop_value = prop
    filename_train = "../data/bankmarketing/bank-full.csv"
    names = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y"
    ]
    df_tr = pd.read_csv(filename_train, names=names, sep=";", skiprows=1)

    # Separate Labels from inputs
    df_tr["y"] = df_tr["y"].astype("category")
    cat_columns = df_tr.select_dtypes(["category"]).columns
    df_tr[cat_columns] = df_tr[cat_columns].apply(lambda x: x.cat.codes)

    cont_cols = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous"
    ]

    cat_cols = [
        "month",
        "contact",
        "marital",
        "job",
        "education",
        "default",
        "housing",
        "loan",
        "poutcome"
    ] 

    df = df_tr
    for col in cat_cols:
        df[col] = df[col].astype("category")

    if custom_balance == True:
        df_tr = generate_class_imbalance(
            data=df_tr, target_class=target_class, target_ratio=target_ratio
        )
    
    if one_hot == False:
        return df_tr

    else:
        df_cat = pd.get_dummies(df[cat_cols])
        # Normalizing continuous coloumns between 0 and 1
        df_cont = df[cont_cols] / (df[cont_cols].max())
        df_cont = df_cont.round(3)

        data = pd.concat([df_cat, df_cont, df["y"]], axis=1)
        prop_key = '%s_%s' % (prop_name, prop_value)
        prop = data[prop_key]

        columns = list(data.columns)
        columns.remove(prop_key)  
        new_order = [prop_key] + columns  
        data = data[new_order]

        return data, prop

def get_bankmk_dataset(property): # [name, value]
    prop_list = {
        'month': 'may', # 0.30
        'marital': 'married', # 0.60
        'contact': 'telephone' # 0.06
    }

    df, prop = load_bankmk_data(one_hot=True, prop=[property, prop_list[property]])
    df = df.reset_index(drop=True)
    y_data = df['y'].to_numpy()
    df = df.drop(['y'], axis=1)
    x_data = df.to_numpy() # (45211, 51)
    prop = prop.to_numpy()
    data = tensor_data_create(x_data, y_data)

    print("Percent of positive classes: {:.2%}".format(np.mean(y_data)))
    print("Percent of target property {}={}: {:.2%}".format(property, prop_list[property], np.mean(prop)))

    return list(data), list(prop)


# if __name__ == '__main__':
#     df1, df2 = get_bankmk_dataset('contact')
#     get_bankmk_dataset('marital')
#     get_bankmk_dataset('month')

# Percent of positive classes: 11.70%
# Percent of target property month=may: 30.45%
# Percent of target property marital=married: 60.19%
# Percent of target property contact=telephone: 6.43%