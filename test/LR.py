# Jiayu Liu
# linear Regression model

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
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

# data standard scalar
ss = StandardScaler()
train_x = pd.DataFrame(ss.fit_transform(train_x))
test_x = pd.DataFrame(ss.transform(test_x))


# define classification score rule
def acc_score(true_value, predict):
    predict[predict < 0.5] = 0
    predict[predict > 0.5] = 1
    return predict[predict == true_value].size / predict.size

# using 5-fold cross valid evaluate test data
scoring = {'acc_score': make_scorer(acc_score, greater_is_better=True)}
clf = LinearRegression()
score, mean, std = [], [], []
for i in range(24):
    clf.fit(train_x, train_y.iloc[:, i])
    cv_cross = cross_validate(clf, test_x, test_y.iloc[:, i], cv=KFold(5), scoring=scoring)
    score.append(cv_cross['test_acc_score'])
    mean.append(cv_cross['test_acc_score'].mean())
    std.append(cv_cross['test_acc_score'].std())

print(np.array(score), np.array(mean), np.array(std))
