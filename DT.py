# Jiayu Liu
# Decision Tree model

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

# read dataset
file = './featured_data.csv'
df = pd.read_csv(file, nrows=70000)
train_x = df.iloc[:65000, 2:20]
train_y = df.iloc[:65000, 20:]
test_x = df.iloc[65000:, 2:20]
test_y = df.iloc[65000:, 20:]

# the relative importance of features
model = ExtraTreesClassifier()
model.fit(train_x, train_y)
importance = model.feature_importances_

features = ['ncodpers', 'sexo', 'age', 'antiguedad', 'tiprel_1mes', 'indext', 'cod_prov', 'ind_actividad_cliente',
            'renta', 'segmento']
train_x = train_x.loc[:, features]
test_x = test_x.loc[:, features]

# grid search best max depth
param = {'max_depth': [i for i in range(2, 9)]}
grid = GridSearchCV(DecisionTreeClassifier(criterion='gini', splitter='best'), param_grid=param, cv=5)
best_param, best_score, score, mean, std = [], [], [], [], []

# deal with multi-class problem
# grid.fit(train_x, train_y)
# print(grid.best_params_)
# print(grid.best_score_)

for i in range(24):
    grid.fit(train_x, train_y.iloc[:, i])
    best_param.append(grid.best_params_['max_depth'])
    best_score.append(grid.best_score_)
    # best model
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=grid.best_params_['max_depth'])
    clf.fit(train_x, train_y.iloc[:, i])
    # evaluate using 5-fold cross valid
    cv_cross = cross_validate(clf, test_x, test_y.iloc[:, i], cv=KFold(5))
    mean.append(cv_cross['test_score'].mean())
    std.append(cv_cross['test_score'].std())
    plot_tree(clf)

# print(best_param)
# print(best_score)
# print(mean)
# print(std)
