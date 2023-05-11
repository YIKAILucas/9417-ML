import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# HAN YIKAI
"""
This part is feature engineering. We encode some string features and extract useful features.
"""
rm_cols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
           'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
           'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
           'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
           'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
           'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
           'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
           'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

targetcols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
              'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
              'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
              'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
              'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
              'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
              'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
              'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

# pd.options.display.max_rows = 500
# np.set_printoptions(threshold=500)

"""
Sampling according to sample proportion, because the current account is most frequently samples,
and it lead to the number of samples to be unbalanced. 
"""


def sample_pro(df):
    new = df.iloc[0:1]
    for i in targetcols:
        if i == 'ind_cco_fin_ult1':
            #         pass
            continue
        tmp = df.loc[df[i] == 1]
        tmp = tmp[0:50000]
        new = new.append(tmp, ignore_index=True)
    # new.shape
    return df


"""
encode label and category features
"""


def process(data):
    encoder_label = LabelEncoder()
    encoder_ord = OrdinalEncoder()

    data.loc[data['sexo'].isnull(), 'sexo'] = "NA"
    # remove 3 unuseful or unapplicable features in our models
    # print(f'rm fecha_alta, canal_entrada, fecha_dato')
    data.drop(['fecha_alta', 'canal_entrada',
               'fecha_dato'], axis=1, inplace=True)

    # to be encoded labels
    to_encode_colum_l = ['sexo', 'ind_nuevo', 'indresi', 'indext', 'indfall',
                         'tiprel_1mes', 'ind_empleado', 'indrel_1mes', 'nomprov', 'pais_residencia']
    # use encoder_label to encode the label features in alphabetical order
    for i in to_encode_colum_l:
        tmp = encoder_label.fit_transform(
            list(data[i].values)).reshape(-1, 1)
        data[i] = tmp.copy()

    return data


def feature_extract(file):
    limit_rows = 500000
    df = pd.read_csv(file,
                     dtype={"sexo": str,
                            "ind_nuevo": str,
                            "ult_fec_cli_1t": str,
                            "indext": str},
                     #                  nrows=limit_rows,
                     low_memory=False)

    df = df.sample(frac=1)
    print(f'initial.shape {df.shape}')
    data = process(df)
    print(f'process data.shape {data.shape}')

    return data


def split_label(data):
    train_data = data.drop(rm_cols, axis=1)
    train_label = data[targetcols]

    return train_data, train_label


def split_train_and_test(train_set, test_set):
    train_X, test_X, train_y, test_y = train_test_split(train_set, test_set, test_size=0.5)

    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    file = "./processed_data.csv"
    data = feature_extract(file)
    data.to_csv('./featured_data.csv')
    # data.to_csv('./featured_data.csv')
    # for model
    # split train and test dataset 
    # train_set, test_set = split_label(data)
    # train_x, train_y, test_x, test_y = split_train_and_test(train_set, test_set)
