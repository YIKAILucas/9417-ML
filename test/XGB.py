from feature_eng import split_label, split_train_and_test
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

# HAN YIKAI
# pd.options.display.max_rows = 500

# np.set_printoptions(threshold=500)

feature4boost = ['ind_empleado', 'pais_residencia', 'sexo', 'age', 'ind_nuevo',
                 'antiguedad', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'indfall', 'cod_prov',
                 'nomprov', 'ind_actividad_cliente', 'renta']

feature4boost = ['tiprel_1mes', 'antiguedad',
                 'age', 'nomprov', 'renta']


def select_features(model):
    features = []
    thresh_hold = 0.15
    for r in range(2, 40, 2):
        #     print(i)
        importance = model.estimators_[r].feature_importances_

        index_num = np.argsort(importance)[::-1]

        for f in range(train_x_4_boost.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, feature4boost[index_num[f]], importance[index_num[f]]))
            if importance[index_num[f]] >= thresh_hold:
                features.append(feature4boost[index_num[f]])
        print(f'features selected {features}')

    return features


def plot_result(x, y, name):
    plt.cla()
    plt.xlabel('number of estimators', fontsize=12)
    plt.ylabel(name, fontsize=12)
    fig = sns.lineplot(x=x, y=y)
    # plt.show()
    pic = fig.get_figure()

    pic.savefig(f'{name}.png', dpi=200)


file = './input/featured_data.csv'
df = pd.read_csv(file, low_memory=False)
df = df.sample(n=50000)
# load data
train_set, test_set = split_label(df)
train_x, train_y, test_x, test_y = split_train_and_test(train_set, test_set)

train_x_4_boost = train_x[feature4boost]
test_x_4_boost = test_x[feature4boost]
print(train_x_4_boost.shape)
l_acc = []
l_hamming = []
n_estimator = list(range(50, 300, 25))
for i in n_estimator:
    clf_multi_xg = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=i,
            base_score=0.8,
            booster='gbtree',
            colsample_bytree=1,
            gamma=0,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            n_jobs=-1,
            num_class=24,
            objective='multi:softmax',
            # objective='multi:softprob',
            reg_alpha=1,
            # reg_lambda=1,
            # scale_pos_weight=1,
            subsample=1,
            use_label_encoder=False))
    # clf_multiclass = XGBClassifier()

    # estimators = list(range(600, 601, 100))
    # param_grid = {'n_estimators': estimators}
    # grid_search = GridSearchCV(
    #     estimator=clf_multi_xg,
    #     param_grid=param_grid,
    #     cv=3,
    #     # n_jobs=-1
    # )
    # training
    print('Training....')
    clf_multi_xg.fit(train_x_4_boost, train_y)
    # predict
    predict_v = clf_multi_xg.predict(test_x_4_boost)
    # use pandas to form result matrix
    res = pd.DataFrame(predict_v)

    # use
    acc = accuracy_score(test_y, predict_v)
    print(f'acc ->:{acc}')

    # use hamming loss to evaluate model
    h_loss = hamming_loss(test_y, predict_v)
    print(f'hamming_loss is : {h_loss}')
    l_acc.append(acc)
    l_hamming.append(h_loss)
print(f'l_acc {l_acc}')
print(f'l_hamming {l_hamming}')

plot_result(n_estimator, l_acc, name='accuracy')
plot_result(n_estimator, l_hamming, name='hamming loss')
# select useful features
# features = select_features(clf_multi_xg)
# ar = grid_search.cv_results_['mean_test_score']
# print(f'cv {ar}')
