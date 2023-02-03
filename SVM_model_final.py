import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# YUCHENG LU takes charge of this final SVC model with kernel='poly', C=0.001, degree=1
# this code will take about 5 minutes to run it
# read training data and set training feature and classification
totalData = pd.read_csv('featured_data.csv')
trainFeature = ['ncodpers', 'ind_empleado', 'sexo', 'age', 'ind_nuevo', 'indrel', 'indrel_1mes',
                'tiprel_1mes', 'indresi', 'indext', 'indfall', 'renta']
trainClassification = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                       'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                       'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                       'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                       'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

# randomly take out rows for training data and test data
np.random.seed(10)
trainRow = np.random.randint(low=0, high=3348634, size=40000)
testRow = np.random.randint(low=0, high=3348634, size=20000)

# read training data and test data, and split data into X and y
trainData = totalData.iloc[trainRow, :]
trainX = trainData.loc[:, trainFeature]
testData = totalData.iloc[testRow, :]
testX = testData.loc[:, trainFeature]

# build the final model and test accuracy using confusion matrix, precision and recall results
for category in trainClassification:
    trainY = trainData.loc[:, [category]]
    numOfClass = len(trainY[category].unique())
    numOfOne = len(trainY[trainY[category] == 1])
    numOfZero = len(trainY[trainY[category] == 0])
    # when current category has two values and the amount of each value is larger than 5, fit the model
    if numOfClass > 1 and numOfOne > 5 and numOfZero > 5:
        finalCLF = SVC(kernel='poly', C=0.001, degree=1)
        finalCLF.fit(trainX, trainY.values.ravel())
        print(category)
        print(finalCLF.score(trainX, trainY))
        # test the model and print confusion matrix, precision and recall results
        testY = testData.loc[:, [category]]
        predictY = finalCLF.predict(testX)
        confusionMatrix = confusion_matrix(testY, predictY)
        classificationReport = classification_report(testY, predictY, zero_division=0)
        print(confusionMatrix)
        print(classificationReport)
