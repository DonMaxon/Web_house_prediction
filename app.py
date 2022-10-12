import sys
sys.path.append('C:\\Users\Max\AppData\Local\Programs\Python\Python310\Lib\site-packages')

from flask import Flask, render_template, request
from statistics import LinearRegression
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects import postgresql
import numpy as np
import sklearn
import math
import imblearn
import collections
import pandas as pd
import scipy

from sqlalchemy.types import ARRAY
from scipy import sparse
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
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://root:root@localhost:5432/house_analisys'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class MlScaler(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    copy = db.Column(db.Boolean)
    mean = db.Column(ARRAY(db.Float, dimensions=1))
    n_features = db.Column(db.Integer)
    n_samples = db.Column(db.Integer)
    scale = db.Column(ARRAY(db.Float, dimensions=1))
    var = db.Column(ARRAY(db.Float, dimensions=1))
    with_mean = db.Column(db.Boolean)
    with_std = db.Column(db.Boolean)

class MlModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    alpha = db.Column(db.Integer)
    coeffs = db.Column(ARRAY(db.Float, dimensions=2))
    copy_x = db.Column(db.Boolean)
    dual_gap = db.Column(db.Float)
    fit_intercept = db.Column(db.Boolean)
    intercept = db.Column(ARRAY(db.Float, dimensions=1))
    l1_ration = db.Column(db.Float)
    max_iter = db.Column(db.Integer)
    n_features_in = db.Column(db.Integer)
    normalize = db.Column(db.String)
    positive = db.Column(db.Boolean)
    precompute = db.Column(db.Boolean)
    random_state = db.Column(db.Integer)
    selection = db.Column(db.String)
    tol = db.Column(db.Float)
    warm_start = db.Column(db.Boolean)
    # sparse_coef = db.Column(ARRAY(db.Float, dimensions=1))


def parseData():
    #d = getConvertFuncs()
    df_rent = pd.read_csv("D:\PythonProjects\IndianHousePrices\House_Rent_Dataset.csv")
    df_data = df_rent[['BHK', 'Size','City', 'Furnishing Status', 'Tenant Preferred', 'Bathroom', 'Point of Contact']]
    df_labels = df_rent[['Rent']]
    df_data=pd.get_dummies(df_data)
    print(df_data.columns)
    cols = df_data.columns
    return df_data.to_numpy(), df_labels.to_numpy(), cols

def splitData():
    data, labels, cols = parseData()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.005)
    df = pd.DataFrame(x_test, columns=cols)
    with open('houses.txt', 'w') as f:
        print(df.to_string(), 'houses.txt', file=f)  # Python 3.x
        print(np.array2string(y_test), 'houses.txt', file=f)
    return x_train, x_test, y_train, y_test, cols

def polynomialRegressionScoreWithLR(x_train, y_train, degree):
    PR = sklearn.preprocessing.PolynomialFeatures(degree=degree)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_train=PR.fit_transform(x_train, y_train)
    LR = sklearn.linear_model.Lasso(alpha=5, max_iter=500000)
    LR.fit(x_train, y_train)
    return LR, scaler, PR

def make_data(num_of_rooms, size, num_of_baths, furnishing_status, city, tenant_preferred, point_of_contact, area_type):
    cities = {
        'Bangalore': [1, 0, 0, 0, 0, 0],
        'Chennai': [0, 1, 0, 0, 0, 0],
        'Delhi': [0, 0, 1, 0, 0, 0],
        'Hyderabad': [0, 0, 0, 1, 0, 0],
        'Kolkata': [0, 0, 0, 0, 1, 0],
        'Mumbai': [0, 0, 0, 0, 0, 1]
    }
    furnish_statuses = {
        'Furnished': [1, 0, 0],
        'Semi-furnished': [0, 1, 0],
        'Unfurnished': [0, 0, 1]
    }
    tenants = {
        'Bachelors': [1, 0, 0],
        'Bachelors/Family': [0, 1, 0],
        'Family': [0, 0, 1]
    }
    contacts = {
        'Contact agent': [1, 0, 0],
        'Contact builder': [0, 1, 0],
        'Contact owner': [0, 0, 1]
    }
    city = np.array(cities[city])
    furnishing_status = np.array(furnish_statuses[furnishing_status])
    tenant_preferred = np.array(tenants[tenant_preferred])
    point_of_contact = np.array(contacts[point_of_contact])
    res = np.concatenate((np.array([num_of_rooms]), np.array([size]),
                         np.array([num_of_baths]), city, furnishing_status, tenant_preferred, point_of_contact))
    return res

x_train = x_test = y_train = y_test =  cols = None
x_train, x_test, y_train, y_test, cols = splitData()

@app.route('/')
def main_page():
    return render_template("index.html")

# @app.route('/action', methods = ['POST', 'GET'])
# def action():
#     if request.method=='POST':
#         num_of_rooms = int(request.form['num_of_rooms'])
#         size = int(request.form['size'])
#         num_of_baths = int(request.form['num_of_baths'])
#         furnishing_status = request.form['furnishing_status']
#         city = request.form['city']
#         tenant_preferred = request.form['tenant_preferred']
#         point_of_contact = request.form['point_of_contact']
#         area_type = request.form['area_type']
#         data = make_data(num_of_rooms, size, num_of_baths, furnishing_status, city, tenant_preferred, point_of_contact, area_type)
#         data = np.reshape(data, (-1, data.size))
#         LR, scaler, PR = polynomialRegressionScoreWithLR(x_train, y_train, 2)
#         data = scaler.transform(data)
#         data = PR.transform(data)
#         pred = LR.predict(data)
#         value = pred[0]
#         print(value)
#         return render_template('value is'+str(value))

def makeLasso(model: MlModel):
    LR = sklearn.linear_model.Lasso()
    LR.coef_=np.array(model.coeffs)
    LR.alpha=model.alpha
    LR.tol=model.tol
    LR.copy_X=model.copy_x
    LR.intercept_=model.intercept
    LR.fit_intercept=np.array(model.intercept)
    LR.dual_gap_=model.dual_gap
    LR.l1_ratio=model.l1_ration
    LR.max_iter=model.max_iter
    LR.n_features_in_=model.n_features_in
    LR.normalize=model.normalize
    LR.positive=model.positive
    LR.precompute=model.precompute
    LR.random_state=model.random_state
    LR.selection=model.random_state
    LR.warm_start=model.warm_start
    # a = np.array(model.sparse_coef, dtype=np.float64)
    # row, col = np.where(a != 0)
    # cellValue = np.array([a[i][j] for i, j in zip(row, col)])
    # a = sparse.csr_matrix((cellValue, (row, col)), shape=(a.shape[0], a.shape[1]), dtype=np.float64)
    # LR.sparse_coef_= a
    return LR

def makeScaler(scaler: MlScaler):
    newScaler = preprocessing.StandardScaler()
    newScaler.copy=scaler.copy
    newScaler.mean_=np.array(scaler.mean)
    newScaler.n_features_in_=scaler.n_features
    newScaler.n_samples_seen_=scaler.n_samples
    newScaler.scale_=np.array(scaler.scale)
    newScaler.var_=np.array(scaler.var)
    newScaler.with_mean=scaler.with_mean
    newScaler.with_std=scaler.with_std
    return newScaler

@app.route('/action', methods = ['POST', 'GET'])
def check_extraction_model():
    if request.method=='POST':
        num_of_rooms = int(request.form['num_of_rooms'])
        size = int(request.form['size'])
        num_of_baths = int(request.form['num_of_baths'])
        furnishing_status = request.form['furnishing_status']
        city = request.form['city']
        tenant_preferred = request.form['tenant_preferred']
        point_of_contact = request.form['point_of_contact']
        area_type = request.form['area_type']
        data = make_data(num_of_rooms, size, num_of_baths, furnishing_status, city, tenant_preferred, point_of_contact, area_type)
        data = np.reshape(data, (-1, data.size))
        model = db.session.query(MlModel).filter(MlModel.id>0).one()
        newLR=makeLasso(model)
        scaler = db.session.query(MlScaler).filter(MlScaler.id > 0).one()
        scaler = makeScaler(scaler)
        data = scaler.transform(data)
        PR = sklearn.preprocessing.PolynomialFeatures(degree=2)
        data = PR.fit_transform(data)
        pred = newLR.predict(data)
        value = pred[0]
        print(value)
        return render_template('value is'+str(value))

# @app.route('/action', methods = ['POST', 'GET'])
# def save_model():
#     if request.method=='POST':
#         num_of_rooms = int(request.form['num_of_rooms'])
#         size = int(request.form['size'])
#         num_of_baths = int(request.form['num_of_baths'])
#         furnishing_status = request.form['furnishing_status']
#         city = request.form['city']
#         tenant_preferred = request.form['tenant_preferred']
#         point_of_contact = request.form['point_of_contact']
#         area_type = request.form['area_type']
#         data = make_data(num_of_rooms, size, num_of_baths, furnishing_status, city, tenant_preferred, point_of_contact, area_type)
#         data = np.reshape(data, (-1, data.size))
#         LR, scaler, PR = polynomialRegressionScoreWithLR(x_train, y_train, 2)
#         # i = db.session.query(MlModel).filter(MlModel.id == 1).one()
#         # db.session.delete(i)
#         # db.session.commit()
#         #MlModel.__table__.drop()
#         sparse = LR.sparse_coef_.toarray().tolist()
#         coefs = LR.coef_.tolist()
#         intercept = LR.intercept_.tolist()
#         saved_scaler = MlScaler(
#             copy=scaler.copy,
#             mean=scaler.mean_,
#             n_features=scaler.n_features_in_,
#             n_samples = int(scaler.n_samples_seen_),
#             scale = scaler.scale_,
#             var = scaler.var_,
#             with_mean = scaler.with_mean,
#             with_std = scaler.with_std
#         )
#         db.session.add(saved_scaler)
#         db.session.commit()
#         model = MlModel(alpha = LR.alpha,
#     coeffs = [coefs],
#     copy_x = LR.copy_X,
#     dual_gap = LR.dual_gap_,
#     fit_intercept = LR.fit_intercept,
#     intercept = intercept,
#     l1_ration = LR.l1_ratio,
#     max_iter = LR.max_iter,
#     n_features_in = LR.n_features_in_,
#     normalize = LR.normalize,
#     positive = LR.positive,
#     precompute = LR.precompute,
#     random_state = LR.random_state,
#     selection = LR.selection,
#     tol = LR.tol,
#     warm_start = LR.warm_start)
#     # sparse_coef = sparse)
#         db.session.add(model)
#         db.session.commit()
#         old_data = data
#         data = scaler.transform(data)
#         data = PR.transform(data)
#         pred = LR.predict(data)
#         value = pred[0]
#         print(value)
#         model = db.session.query(MlModel).filter(MlModel.id>0).one()
#         scaler = db.session.query(MlScaler).filter(MlScaler.id > 0).one()
#         newLR=makeLasso(model)
#         scaler=makeScaler(scaler)
#         data = scaler.transform(old_data)
#         data = PR.transform(data)
#         pred = newLR.predict(data)
#         value = pred[0]
#         print(value)
#         return render_template('value is'+str(value))

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run()
