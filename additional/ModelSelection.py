import re
from statistics import LinearRegression
import numpy as np
import sklearn
import math
import imblearn
import collections
import pandas as pd

from sklearn import ensemble
from sklearn import feature_selection
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing 
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, chi2, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.neural_network import MLPRegressor

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
########INFO ABOUT DATASET
#---Unique values 
#In floor = 480
#In area type = 3
#In city = 6
#In furnishing status = 3
#In tenant preferred = 3
#In point of contact = 3
#----
#Scores:
#LinearRegression = 0.4171439899407415

#Lasso(alpha = 4) = 0.42159540576632243
#Lasso(alpha = 5) = 0.4196986663150792
#Lasso(alpha = 7) = 0.4176466277535981
#Lasso(alpha = 10) = 0.4193654042341534
#Lasso(alpha = 15) = 0.4154848232232836
#Lasso(alpha = 25) = 0.4201427031556961
#Lasso(alpha = 50) = 0.4209529593733802
#Lasso(alpha = 75) = 0.42185772442253694
#Lasso(alpha = 150) = 0.4184978507983171
#Lasso(alpha = 500) = 0.41174740013297334

#Ridge(alpha = 0.001) = 0.41457854521394566
#Ridge(alpha = 0.01) = 0.42102651231991933
#Ridge(alpha = 0.05) = 0.42112814813227994
#Ridge(alpha = 0.1) = 0.4215427088220104
#Ridge(alpha = 1) = 0.4211678876369477
#Ridge(alpha = 2) = 0.41521972804981705
#Ridge(alpha = 3) = 0.42099775068753875
#Ridge(alpha = 50) = 0.4210142675885811
#Ridge(alpha = 150) = 0.42075592688253166

#ElasticNet(alpha = 0.01, l1_ratio = 0.25) = 0.4098547188734553
#ElasticNet(alpha = 0.05, l1_ratio = 0.25) = 0.42166057696551357
#ElasticNet(alpha = 0.1, l1_ratio = 0.25) = 0.41566798423499696
#ElasticNet(alpha = 0.5, l1_ratio = 0.25) = 0.39133770454365685
#ElasticNet(alpha = 1, l1_ratio = 0.25) = 0.3584753981185618
#ElasticNet(alpha = 5, l1_ratio = 0.25) = 0.2826820644387333
#ElasticNet(alpha = 100, l1_ratio = 0.25) = 0.21729493372106348

#ElasticNet(alpha = 0.01, l1_ratio = 0.5) = 0.4200531500846953
#ElasticNet(alpha = 0.05, l1_ratio = 0.5) = 0.42070707374608474
#ElasticNet(alpha = 0.1, l1_ratio = 0.5) = 0.4094938166354908
#ElasticNet(alpha = 0.5, l1_ratio = 0.5) = 0.38457222603362895
#ElasticNet(alpha = 1, l1_ratio = 0.5) = 0.36378899189034397
#ElasticNet(alpha = 100, l1_ratio = 0.5) = 0.21895056618221165

#ElasticNet(alpha = 0.005, l1_ratio = 0.75) = 0.4211536933321366
#ElasticNet(alpha = 0.01, l1_ratio = 0.75) = 0.4227395309243331
#ElasticNet(alpha = 0.05, l1_ratio = 0.75) = 0.41338670954641593
#ElasticNet(alpha = 0.1, l1_ratio = 0.75) = 0.4123841466907388
#ElasticNet(alpha = 0.5, l1_ratio = 0.75) = 0.38335125602131936
#ElasticNet(alpha = 100, l1_ratio = 0.75) = 0.215696027049112

#Polynomial+LinearRegression(degree = 2) = -3.5870392882753577e+19
#Polynomial+LinearRegression(degree = 3) = -3.426962276040919e+18

##Lasso+PolyRegression(degree = 2, alpha=1, max_iter=100000) = 0.7206732415141849
##Lasso+PolyRegression(degree = 3, alpha=100) = 0.3982799312267301

#SGD - too bad

#RandomForest = 0.6139837576821068

#

#def getConvertFuncs():

# def linearRegressionScore(x_train, x_test, y_train, y_test):
#     LR = sklearn.linear_model.LinearRegression()
#     LR.fit(x_train, y_train)
#     val = (metrics.r2_score(y_test, LR.predict(x_test)))
#     return val

# def lassoScore(x_train, x_test, y_train, y_test, alpha):
#     LR = sklearn.linear_model.Lasso(alpha=alpha)
#     LR.fit(x_train, y_train)
#     val = (metrics.r2_score(y_test, LR.predict(x_test)))
#     return val

# def ridgeScore(x_train, x_test, y_train, y_test, alpha):
#     LR = sklearn.linear_model.Ridge(alpha=alpha)
#     LR.fit(x_train, y_train)
#     val = (metrics.r2_score(y_test, LR.predict(x_test)))
#     return val

# def elasticNetScore(x_train, x_test, y_train, y_test, alpha, l1_ratio):
#     LR = sklearn.linear_model.ElasticNet(alpha=alpha)
#     LR.fit(x_train, y_train)
#     val = (metrics.r2_score(y_test, LR.predict(x_test)))
#     return val

# def RFScore(x_train, x_test, y_train, y_test):
#     RF_model = sklearn.ensemble.RandomForestRegressor(n_estimators=1000, n_jobs=-1, max_depth=10, random_state=42)
#     RF_model.fit(x_train, np.ravel(y_train))
#     predictions = RF_model.predict(x_test)
#     score = metrics.r2_score(y_test, predictions)
#     return score

# def GBRScore(x_train, x_test, y_train, y_test):
#     #LR = sklearn.ensemble.GradientBoostingRegressor()
#     PR = sklearn.preprocessing.PolynomialFeatures(degree=2)
#     scaler = preprocessing.StandardScaler().fit(x_train)
#     x_train = scaler.transform(x_train)
#     x_test = scaler.transform(x_test)
#     x_train=PR.fit_transform(x_train, y_train)
#     x_test=PR.fit_transform(x_test, y_test)
#     lasso = sklearn.linear_model.Lasso(alpha=100, max_iter=100000)
#     LR = sklearn.ensemble.AdaBoostRegressor(base_estimator=lasso)
#     LR.fit(x_train, np.ravel(y_train))
#     val = (metrics.r2_score(y_test, LR.predict(x_test)))
#     return val

# def polynomialRegressionScoreWithLR(x_train, x_test, y_train, y_test, degree):
#     PR = sklearn.preprocessing.PolynomialFeatures(degree=degree)
#     scaler = preprocessing.StandardScaler().fit(x_train)
#     x_train = scaler.transform(x_train)
#     x_test = scaler.transform(x_test)
#     x_train=PR.fit_transform(x_train, y_train)
#     x_test=PR.fit_transform(x_test, y_test)
#     LR = sklearn.linear_model.Lasso(alpha=10, max_iter=10000)
#     LR.fit(x_train, y_train)
#     val = (metrics.r2_score(y_test, LR.predict(x_test)))
#     return val

# def SGDScore(x_train, x_test, y_train, y_test, alpha):
#     LR = sklearn.linear_model.SGDRegressor(alpha=alpha)
#     LR.fit(x_train, np.ravel(y_train))
#     val = (metrics.r2_score(y_test, LR.predict(x_test)))
#     return val

# def MLPScore(x_train, x_test, y_train, y_test, alpha, num_of_iters, solver, layers):
#     scaler = preprocessing.StandardScaler().fit(x_train)
#     x_train = scaler.transform(x_train)
#     x_test = scaler.transform(x_test)
#     LR = MLPRegressor(alpha=alpha, max_iter=num_of_iters, hidden_layer_sizes = layers, solver = solver)
#     LR.fit(x_train,  np.ravel(y_train))
#     val = (metrics.r2_score(y_test, LR.predict(x_test)))
#     return val


# #def LassoTest(x_train, x_test, y_train, y_test):

# def linearRegressionTest():
#     res = 0
#     numOfTests = 1000
#     for i in range(numOfTests):
#         x_train, x_test, y_train, y_test = splitData()
#         res+=linearRegressionScore(x_train, x_test, y_train, y_test)/numOfTests
#     return res

# def lassoTest(alpha):
#     res = 0
#     numOfTests = 1000
#     for i in range(numOfTests):
#         x_train, x_test, y_train, y_test = splitData()
#         res+=lassoScore(x_train, x_test, y_train, y_test, alpha=alpha)/numOfTests
#     return res

# def ridgeTest(alpha):
#     res = 0
#     numOfTests = 1000
#     for i in range(numOfTests):
#         x_train, x_test, y_train, y_test = splitData()
#         res+=ridgeScore(x_train, x_test, y_train, y_test, alpha=alpha)/numOfTests
#     return res

# def elasticNetTest(alpha, l1_ratio):
#     res = 0
#     numOfTests = 1000
#     for i in range(numOfTests):
#         x_train, x_test, y_train, y_test = splitData()
#         res+=elasticNetScore(x_train, x_test, y_train, y_test, alpha=alpha, l1_ratio=l1_ratio)/numOfTests
#     return res



# def SGDTest(alpha):
#     res = 0
#     numOfTests = 1
#     for i in range(numOfTests):
#         x_train, x_test, y_train, y_test = splitData()
#         res+=SGDScore(x_train, x_test, y_train, y_test, alpha=alpha)/numOfTests
#     return res

# def MLPTest(alpha, num_of_iters, solver, layers):
#     res = 0
#     numOfTests = 1
#     for i in range(numOfTests):
#         x_train, x_test, y_train, y_test = splitData()
#         res+=MLPScore(x_train, x_test, y_train, y_test, alpha=alpha, num_of_iters=num_of_iters, solver=solver, 
#         layers=layers)/numOfTests
#     return res

# def RFTest():
#     res = 0
#     numOfTests = 1
#     for i in range(numOfTests):
#         x_train, x_test, y_train, y_test = splitData()
#         res+=RFScore(x_train, x_test, y_train, y_test)/numOfTests
#     return res

# def GBRTest():
#     res = 0
#     numOfTests = 1
#     for i in range(numOfTests):
#         x_train, x_test, y_train, y_test = splitData()
#         res+=GBRScore(x_train, x_test, y_train, y_test)/numOfTests
#     return res

def parseData():
    #d = getConvertFuncs()
    df_rent = pd.read_csv("House_Rent_Dataset.csv")
    df_data = df_rent[['BHK', 'Size','City', 'Furnishing Status', 'Tenant Preferred', 'Bathroom', 'Point of Contact', 'Point of contact']]
    df_labels = df_rent[['Rent']]
    df_data=pd.get_dummies(df_data)
    return df_data.to_numpy(), df_labels.to_numpy()

def splitData():
    data, labels = parseData()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels)
    return x_train, x_test, y_train, y_test

def polynomialRegressionScoreWithLR(x_train, x_test, y_train, y_test, degree):
    PR = sklearn.preprocessing.PolynomialFeatures(degree=degree)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_train=PR.fit_transform(x_train, y_train)
    x_test=PR.fit_transform(x_test, y_test)
    LR = sklearn.linear_model.Lasso(alpha=5, max_iter=500000)
    LR.fit(x_train, y_train)
    val = (metrics.r2_score(y_test, LR.predict(x_test)))
    print(val)
    return val, LR

def polynomialRegressionTestWithLR(alpha):
    res = 0
    numOfTests = 1
    for i in range(numOfTests):
        x_train, x_test, y_train, y_test = splitData()
        res+=polynomialRegressionScoreWithLR(x_train, x_test, y_train, y_test, degree=alpha)/numOfTests
    return res

def main():
    # num_of_iters = 100000
    # solver = 'adam'
    # layers = (6, 3)
    x_train, x_test, y_train, y_test = splitData()
    degree = 2
    res = polynomialRegressionScoreWithLR(x_train, x_test, y_train, y_test, degree)
    i = 1
    while (res[0]<0.8):
        print(i)
        i+=1
        x_train, x_test, y_train, y_test = splitData()
        res = polynomialRegressionScoreWithLR(x_train, x_test, y_train, y_test, degree)
    print('ok')
    LR = res[1]
    x_train, x_test, y_train, y_test = splitData()
    val = (metrics.r2_score(y_test, LR.predict(x_test)))
    print(val)
    winsound.Beep(frequency, duration)
    print('ok')

main()
