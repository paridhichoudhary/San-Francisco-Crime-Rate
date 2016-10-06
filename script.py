# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:22:10 2016

@author: Paridhi
"""

import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt
from datetime import date
from datetime import time
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif


class DataDigest:

    def __init__(self):
        self.date = None
        self.weekday = None
        self.district = None
        self.resolution = None
        self.X = None
        self.Y = None


def get_date(datestring):
    datestr, timestr = re.split(" ",datestring)
    year,month,day = re.split("-",datestr)
    dat = date(int(year), int(month), int(day))
    return dat
    
def get_time(datestring):
    datestr, timestr = re.split(" ",datestring)
    hour,minute,sec = re.split(":",timestr)
    tim = time(int(hour), int(minute), int(sec))
    return tim
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def munge_data(data, digest):
    # Categories
    category_list = {'ARSON':1,'ASSAULT':2,'BAD CHECKS':3,'BRIBERY':4,'BURGLARY':5,'DISORDERLY CONDUCT':6,'DRIVING UNDER THE INFLUENCE':7,'DRUG/NARCOTIC':8,'DRUNKENNESS':9,'EMBEZZLEMENT':10,'EXTORTION':11,'FAMILY OFFENSES':12,'FORGERY/COUNTERFEITING':13,'FRAUD':14,'GAMBLING':15,'KIDNAPPING':16,'LARCENY/THEFT':17,'LIQUOR LAWS':18,'LOITERING':19,'MISSING PERSON':20,'NON-CRIMINAL':21,'OTHER OFFENSES':22,'PORNOGRAPHY/OBSCENE MAT':23,'PROSTITUTION':24,'RECOVERED VEHICLE':25,'ROBBERY':26,'RUNAWAY':27,'SECONDARY CODES':28,'SEX OFFENSES FORCIBLE':29,'SEX OFFENSES NON FORCIBLE':30,'STOLEN PROPERTY':31,'SUICIDE':32,'SUSPICIOUS OCC':33,'TREA':34,'TRESPASS':35,'VANDALISM':36,'VEHICLE THEFT':37,'WARRANTS':38,'WEAPON LAWS':39}
    data["CategoryF"] = data["Category"].apply(lambda s: category_list.get(s))

    category_dummies = pd.get_dummies(data["Category"], prefix="CategoryD", dummy_na=False)
    data = pd.concat([data, category_dummies], axis=1)

    # weekday
    weekdays = {"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
    data["WeekdayF"] = data["DayOfWeek"].fillna("U").apply(lambda e: weekdays.get(e))

    weekdays_dummies = pd.get_dummies(data["DayOfWeek"], prefix="WeekdayD", dummy_na=False)
    data = pd.concat([data, weekdays_dummies], axis=1)

    # District
    districts =  {'BAYVIEW':1,'CENTRAL':2,'INGLESIDE':3,'MISSION':4,'NORTHERN':5,'PARK':6,'RICHMOND':7,'SOUTHERN':8, 'TARAVAL':9, 'TENDERLOIN':10 }
    data["DistrictF"] = data["PdDistrict"].apply(lambda r: districts.get(r))

    district_dummies = pd.get_dummies(data["PdDistrict"], prefix="DistrictD", dummy_na=False)
    data = pd.concat([data, district_dummies], axis=1)
    
    # Resolution
    resolutions = {"ARREST, BOOKED":1, "ARREST, CITED":2,"CLEARED-CONTACT JUVENILE FOR MORE INFO":3,"COMPLAINANT REFUSES TO PROSECUTE":4,"DISTRICT ATTORNEY REFUSES TO PROSECUTE":5,"EXCEPTIONAL CLEARANCE":6,"JUVENILE ADMONISHED":7,"JUVENILE BOOKED":8,"JUVENILE CITED":9,"JUVENILE DIVERTED":10,"LOCATED":11,"NONE":12,"NOT PROSECUTED":13,"PROSECUTED BY OUTSIDE AGENCY":14,"PROSECUTED FOR LESSER OFFENSE":15,"PSYCHOPATHIC CASE":16,"UNFOUNDED":17}
    data["ResolutionF"] = data["Resolution"].fillna("U").apply(lambda c: resolutions.get(c))

    resolution_dummies = pd.get_dummies(data["Resolution"], prefix="ResolutionD", dummy_na=False)
    data = pd.concat([data, resolution_dummies], axis=1)

    # Distance
    data["DistanceF"] = data.apply(lambda h: haversine(h["X"],h["Y"],digest.X, digest.Y),axis =1)    
    
    # Stat
    distance_bins = [0, 0.5, 1.0,1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10, 100, 1000,2000,3000,4000,5000,6000]
    data["DistanceR"] = pd.cut(data["DistanceF"].fillna(-1), bins=distance_bins).astype(object)
    
    distances = {'(0, 0.5]':0, '(0.5, 1]':0.5, '(1, 1.5]':1, '(1.5, 2]':1.5, '(2, 2.5]':2,
       '(2.5, 3]':2.5, '(3, 3.5]':3, '(3.5, 4]':3.5, '(4, 4.5]':4, '(4.5, 5]':4.5,
       '(5, 10]':5, '(5000, 6000]':5000}
    data['DistanceRR'] = data['DistanceR'].fillna('U').apply(lambda t: distances.get(t))
    
    #Date
    data['DateF'] = data['Dates'].apply(lambda r: get_date(r))
    
    #Time
    data['TimeF'] = data['Dates'].apply(lambda r: get_time(r))

    return data


def linear_scorer(estimator, x, y):
    scorer_predictions = estimator.predict(x)

    scorer_predictions[scorer_predictions > 0.5] = 1
    scorer_predictions[scorer_predictions <= 0.5] = 0

    return metrics.accuracy_score(y, scorer_predictions)

# -----------------------------------------------------------------------------
# load
# -----------------------------------------------------------------------------

train_data = pd.read_csv("D:/Kaggle/SF Crime Rate/train.csv")
test_data = pd.read_csv("D:/Kaggle/SF Crime Rate/test.csv")
all_data = pd.concat([train_data, test_data])

# -----------------------------------------------------------------------------
# stat
# -----------------------------------------------------------------------------

print("===== survived by DayOfWeek and PdDistrict")
print(train_data.groupby(["DayOfWeek", "PdDistrict"])["Category"].value_counts(normalize=True))

# -----------------------------------------------------------------------------
# describe
# -----------------------------------------------------------------------------

describe_fields = ["Category", "DayOfWeek", "PdDistrict", "Resolution", "X", "Y"]

print("===== train: males")

print("===== test: males")
print(test_data[test_data["Sex"] == 0][describe_fields].describe())

print("===== train: females")
print(train_data[train_data["Sex"] == 1][describe_fields].describe())

print("===== test: females")
print(test_data[test_data["Sex"] == 1][describe_fields].describe())

# -----------------------------------------------------------------------------
# munge
# -----------------------------------------------------------------------------

data_digest = DataDigest()

data_digest.X = all_data['X'].mean()
data_digest.Y = all_data['Y'].mean()

#categories_trn = pd.Index(train_data["Category"].unique())
#data_digest.categories = categories_trn 

district_trn = pd.Index(train_data["PdDistrict"].unique())
district_tst = pd.Index(test_data["PdDistrict"].unique())
data_digest.district = district_tst

train_data_munged = munge_data(train_data, data_digest)
test_data_munged = munge_data(test_data, data_digest)
all_data_munged = pd.concat([train_data_munged, test_data_munged])

predictors = ['WeekdayF', 
              'DistrictF',
              'DistanceRR']

cv = StratifiedKFold(train_data_munged["CategoryF"], n_folds=3, shuffle=True, random_state=1)

# -----------------------------------------------------------------------------
# stat 2
# -----------------------------------------------------------------------------

print("===== survived by age")
print(train_data_munged.groupby(["AgeR"])["Survived"].value_counts(normalize=True))

print("===== survived by gender and age")
print(train_data_munged.groupby(["Sex", "AgeR"])["Survived"].value_counts(normalize=True))

print("===== survived by class and age")
print(train_data_munged.groupby(["Pclass", "AgeR"])["Survived"].value_counts(normalize=True))

# -----------------------------------------------------------------------------
# pairplot graph
# -----------------------------------------------------------------------------
sns.pairplot(train_data_munged, vars=["AgeF", "Pclass", "SexF"], hue="Survived", dropna=True)
# sns.plt.show()

# ----------------------------------------------------------------------------
# features graph
# -----------------------------------------------------------------------------

selector = SelectKBest(f_classif, k=5)
selector.fit(train_data_munged[predictors], train_data_munged["CategoryF"])


scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

# -----------------------------------------------------------------------------
# scale
# -----------------------------------------------------------------------------

scaler = StandardScaler()
scaler.fit(all_data_munged[predictors])

# scaled
train_data_scaled = scaler.transform(train_data_munged[predictors])
test_data_scaled = scaler.transform(test_data_munged[predictors])

# non-scaled
# train_data_scaled = train_data_munged[predictors]
# test_data_scaled = test_data_munged[predictors]

# -----------------------------------------------------------------------------
# K-neighbourhood
# -----------------------------------------------------------------------------

alg_ngbh = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(alg_ngbh, train_data_scaled, train_data_munged["CategoryF"], cv=cv, n_jobs=-1)
print("Accuracy (k-neighbors): {}/{}".format(scores.mean(), scores.std()))

# -----------------------------------------------------------------------------
# sgd
# -----------------------------------------------------------------------------

alg_sgd = SGDClassifier(random_state=1)
scores = cross_val_score(alg_sgd, train_data_scaled, train_data_munged["CategoryF"], cv=cv, n_jobs=-1)
print("Accuracy (sgd): {}/{}".format(scores.mean(), scores.std()))

# -----------------------------------------------------------------------------
# svm
# -----------------------------------------------------------------------------

alg_svm = SVC(C=1.0)
scores = cross_val_score(alg_svm, train_data_scaled, train_data_munged["CategoryF"], cv=cv, n_jobs=-1)
print("Accuracy (svm): {}/{}".format(scores.mean(), scores.std()))

# -----------------------------------------------------------------------------
# naive bayes
# -----------------------------------------------------------------------------

alg_nbs = GaussianNB()
scores = cross_val_score(alg_nbs, train_data_scaled, train_data_munged["CategoryF"], cv=cv, n_jobs=-1)
print("Accuracy (naive bayes): {}/{}".format(scores.mean(), scores.std()))

# -----------------------------------------------------------------------------
# linear regression
# -----------------------------------------------------------------------------

alg_lnr = LinearRegression()
scores = cross_val_score(alg_lnr, train_data_scaled, train_data_munged["CategoryF"], cv=cv, n_jobs=-1,
                         scoring=linear_scorer)
print("Accuracy (linear regression): {}/{}".format(scores.mean(), scores.std()))

# -----------------------------------------------------------------------------
# logistic regression
# -----------------------------------------------------------------------------

alg_log = LogisticRegression(random_state=1)
scores = cross_val_score(alg_log, train_data_scaled, train_data_munged["CategoryF"], cv=cv, n_jobs=-1,
                         scoring=linear_scorer)
print("Accuracy (logistic regression): {}/{}".format(scores.mean(), scores.std()))

# -----------------------------------------------------------------------------
# random forest simple
# -----------------------------------------------------------------------------

alg_frst = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=8, min_samples_leaf=2)
scores = cross_val_score(alg_frst, train_data_scaled, train_data_munged["CategoryF"], cv=cv, n_jobs=-1)
print("Accuracy (random forest): {}/{}".format(scores.mean(), scores.std()))

# -----------------------------------------------------------------------------
# random forest auto
# -----------------------------------------------------------------------------

alg_frst_model = RandomForestClassifier(random_state=1)
alg_frst_params = [{
    "n_estimators": [350, 400, 450],
    "min_samples_split": [6, 8, 10],
    "min_samples_leaf": [1, 2, 4]
}]
alg_frst_grid = GridSearchCV(alg_frst_model, alg_frst_params, cv=cv, refit=True, verbose=1, n_jobs=-1)
alg_frst_grid.fit(train_data_scaled, train_data_munged["CategoryF"])
alg_frst_best = alg_frst_grid.best_estimator_
print("Accuracy (random forest auto): {} with params {}"
      .format(alg_frst_grid.best_score_, alg_frst_grid.best_params_))

# -----------------------------------------------------------------------------
# XBoost auto
# -----------------------------------------------------------------------------

ald_xgb_model = xgb.XGBClassifier()
ald_xgb_params = [
    {"n_estimators": [230, 250, 270],
     "max_depth": [1, 2, 4],
     "learning_rate": [0.01, 0.02, 0.05]}
]
alg_xgb_grid = GridSearchCV(ald_xgb_model, ald_xgb_params, cv=cv, refit=True, verbose=1, n_jobs=1)
alg_xgb_grid.fit(train_data_scaled, train_data_munged["CategoryF"])
alg_xgb_best = alg_xgb_grid.best_estimator_
print("Accuracy (xgboost auto): {} with params {}"
      .format(alg_xgb_grid.best_score_, alg_xgb_grid.best_params_))

# -----------------------------------------------------------------------------
# test output
# -----------------------------------------------------------------------------

alg_test = alg_frst_best

alg_test.fit(train_data_scaled, train_data_munged["CategoryF"])

predictions = alg_test.predict(test_data_scaled)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})

submission.to_csv("titanic-submission.csv", index=False)


train_data_munged = pd.read_csv("D:/Kaggle/SF Crime Rate/SF Crime Rate_Submission_random_auto_v1.csv")
test_data_munged = pd.read_csv("D:/Kaggle/SF Crime Rate/SF Crime Rate_test_random_auto_v1.csv")
all_data_munged = pd.concat([train_data_munged, test_data_munged])

distance_bins = [0, 0.5, 1.0,1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10, 100, 1000,2000,3000,4000,5000,6000]
train_data_munged["DistanceR"] = pd.cut(train_data_munged["DistanceF"].fillna(-1), bins=distance_bins).astype(object)


distances = {'(0, 0.5]':0, '(0.5, 1]':0.5, '(1, 1.5]':1, '(1.5, 2]':1.5, '(2, 2.5]':2,
      '(2.5, 3]':2.5, '(3, 3.5]':3, '(3.5, 4]':3.5, '(4, 4.5]':4, '(4.5, 5]':4.5,
      '(5, 10]':5, '(5000, 6000]':5000}
train_data_munged['DistanceRR'] = train_data_munged['DistanceR'].fillna('U').apply(lambda t: distances.get(t))
test_data_munged['DistanceRR'] = test_data_munged['DistanceR'].fillna('U').apply(lambda t: distances.get(t))
