from sklearn.model_selection import GridSearchCV

from feature_eng import split_label, split_train_and_test
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss

# HAN YIKAI
feature4rf = ['ind_empleado', 'pais_residencia', 'sexo', 'age', 'ind_nuevo',
              'antiguedad', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'indfall', 'cod_prov',
              'nomprov', 'ind_actividad_cliente', 'renta']
# feature4rf = ['age',
#               'antiguedad', 'renta']
feature4rf = ["ind_empleado", "pais_residencia", "sexo",
              "age", "ind_nuevo", "antiguedad", "nomprov",
              "segmento"]

file = './input/featured_data.csv'
df = pd.read_csv(file, low_memory=False)
df = df.sample(n=40000)

print(f'df.shape {df.shape}')

train_set, test_set = split_label(df)
train_x, train_y, test_x, test_y = split_train_and_test(train_set, test_set)

train_x_4_rf = train_x[feature4rf]
test_x_4_rf = test_x[feature4rf]

"""
plot result fig to evaluate model performance 
"""


def plot_result(x, y, name):
    plt.cla()
    plt.xlabel('number of min_samples_leaf', fontsize=12)
    plt.ylabel(name, fontsize=12)
    fig = sns.lineplot(x=x, y=y)
    # plt.show()
    pic = fig.get_figure()

    pic.savefig(f'{name}.png', dpi=150)


"""
select useful features with impact
"""


def select_features(model):
    features = []
    thresh_hold = 0.15
    importance = model.feature_importances_
    index_num = np.argsort(importance)[::-1]

    for f in range(train_x_4_rf.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feature4rf[index_num[f]], importance[index_num[f]]))
        if importance[index_num[f]] >= thresh_hold:
            features.append(feature4rf[index_num[f]])
    print(f'features selected {features}')

    return features


n_estimators = list(range(10, 200, 10))
max_depth = list(range(15, 60, 5))
min_weight_fraction_leaf = list(range(0, 10, 2))
param_grid = {'n_estimators': n_estimators,
              'oob_score': [True],
              # 'min_samples_split': [5],
              # 'min_samples_leaf': [30],
            #   'min_weight_fraction_leaf': min_weight_fraction_leaf,
              'max_depth': [20]
              }

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid,
                           cv=3)
# clf = grid_search
# clf = RandomForestClassifier(n_estimators=20,
#                              max_depth=None,
#                              n_jobs=-1,
#                              random_state=2021,
#                              # class_weight='balanced'
#                              )
print('Training....')
grid_search.fit(train_x_4_rf, train_y)
h_loss_l = []
acc_l = []
best_score_l = []
print('Predicting....')
predict_v = grid_search.predict(test_x_4_rf)

print('acc:', accuracy_score(test_y, predict_v))

res = pd.DataFrame(predict_v)

h_loss = hamming_loss(test_y, predict_v)
print(f'hamming_loss is : {h_loss}')
# print(res.iloc[11])

# select features in first time, clf have been annotation
# selected_features = select_features(clf)

mean_test_score = grid_search.cv_results_['mean_test_score']
print(f'cv {mean_test_score}')
print(f'type {type(mean_test_score)}')

# use grid search to determine the best params in this model
print(f'best params {grid_search.best_params_}')
print(f'best score {grid_search.best_score_}')
x_columns_indices = []

# filter features
# x_selected = train_x[:, importances > threshold]
# print(x_selected)

plot_result(n_estimators, mean_test_score, name='accuracy of Random forest')
# plot_result(n_estimators, l_hamming, name='hamming.png')
