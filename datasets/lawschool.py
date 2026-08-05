import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from my_utils.utils import tensor_data_create


# --------------------------------------------------------------------------------------


def load_lawschool_data(one_hot=True, custom_balance=False, target_class=1, target_ratio=0.3, prop=None):
    """Load the bankmk dataset."""

    prop_name, prop_value = prop
    # df_tr = pd.read_csv(filename_train, names=names, sep=";", skiprows=1)
    df_tr = pd.read_stata('../data/Lawschool/lawschs1_1.dta')
    # preprocess
    df_tr.drop(['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'], axis=1, inplace=True)
    df_tr.dropna(axis=0, inplace=True, subset=['admit']) # filter NaN column
    df_tr.replace(to_replace='', value=np.nan, inplace=True)
    df_tr.dropna(axis=0, inplace=True)

    # Separate Labels from inputs
    df_tr["admit"] = df_tr["admit"].astype("category")
    cat_columns = df_tr.select_dtypes(["category"]).columns
    df_tr[cat_columns] = df_tr[cat_columns].apply(lambda x: x.cat.codes)

    cont_cols = [
        "lsat",
        "gpa"
    ]

    cat_cols = [
        "race",
        "resident",
        "college",
        "year",
        "gender"
    ] 

    df = df_tr
    for col in cat_cols:
        df[col] = df[col].astype("category")

    if one_hot == False:
        return df_tr

    else:
        df_cat = pd.get_dummies(df[cat_cols])
        # Normalizing continuous coloumns between 0 and 1
        df_cont = df[cont_cols] / (df[cont_cols].max())
        df_cont = df_cont.round(3)

        data = pd.concat([df_cat, df_cont, df["admit"]], axis=1)

        prop_key = '%s_%s' % (prop_name, prop_value)
        prop = data[prop_key]

        columns = list(data.columns)
        columns.remove(prop_key)  
        new_order = [prop_key] + columns  
        data = data[new_order]

        return data, prop

def get_lawschool_dataset(property): # [name, value]
    prop_list = {
        'race': 'Black', # 8.63%
        'resident': '0.0', # 69.08%
        'gender': '0.0' # 44.27%
    }

    df, prop = load_lawschool_data(one_hot=True, prop=[property, prop_list[property]])
    df = df.reset_index(drop=True)
    y_data = df['admit'].to_numpy()
    df = df.drop(['admit'], axis=1)
    x_data = df.to_numpy()  #(96584, 39)
    prop = prop.to_numpy()
    data = tensor_data_create(x_data, y_data)

    print("Percent of positive classes: {:.2%}".format(np.mean(y_data)))
    print("Percent of target property {}={}: {:.2%}".format(property, prop_list[property], np.mean(prop)))

    return list(data), list(prop)


# if __name__ == '__main__':
#     df1, df2 = get_lawschool_dataset('race')
#     get_lawschool_dataset('resident')
#     get_lawschool_dataset('gender')

    
    # Percent of positive classes: 26.49%
    # Percent of target property race=Black: 8.63%
    # Percent of target property resident=0.0: 69.08%
    # Percent of target property gender=0.0: 44.27%
