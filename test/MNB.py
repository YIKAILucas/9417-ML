# Jiayu Liu
# Multinomial Naive bayes model

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
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
# model = ExtraTreesClassifier()
# model.fit(train_x, train_y)
# importance = model.feature_importances_

features = ['ncodpers', 'sexo', 'age', 'antiguedad', 'tiprel_1mes', 'indext', 'cod_prov', 'ind_actividad_cliente',
            'renta', 'segmento']
train_x = train_x.loc[:, features]
test_x = test_x.loc[:, features]

# grid search best smooth alpha
param = {'alpha': [i / 10 for i in range(0, 11)]}
grid = GridSearchCV(MultinomialNB(), param_grid=param, cv=5)
best_param, score, mean, std = [], [], [], []

for i in range(24):
    grid.fit(train_x, train_y.iloc[:, i])
    best_param.append(grid.best_score_)
    # best model
    clf = MultinomialNB(alpha=grid.best_params_['alpha'])
    # evaluate using 5-fold cross valid
    cv_cross = cross_validate(clf, test_x, test_y.iloc[:, i], cv=KFold(5))
    score.append(cv_cross['test_score'])
    mean.append(cv_cross['test_score'].mean())
    std.append(cv_cross['test_score'].std())

print(best_param)
print(score)
print(mean)
print(std)

