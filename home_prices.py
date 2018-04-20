# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:50:15 2018

@author: kaushik
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', -1)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from math import sqrt
from sklearn.metrics import r2_score
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import lightgbm as lgb 

#read the files and save in dfs
train = pd.read_csv("C:\\Users\\kaushik\\AnacondaProjects\\train-housing.csv")
test = pd.read_csv("C:\\Users\\kaushik\\AnacondaProjects\\test-housing.csv")

#check for duplicates in train
train['Id'].nunique()
train.shape[0]

#check for outliers
_, bp = pd.DataFrame.boxplot(train['GrLivArea'], return_type = 'both')
outliers = [flier.get_ydata() for flier in bp["fliers"]]
outliers#too many outliers

#let's check for outliers in a scatterplot
plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])

#in linear regression, linear outliers are fine, but non-linear outliers should be removed
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

plt.scatter(x=train['GrLivArea'], y = train['SalePrice'])

#We cant remove outliers for all variables, so we can use robustscalar during modelling


#let's examine the output variable
sns.distplot(train['SalePrice'])#the dist seems to be right skewed

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)

#lets perform log operations to make the dist normal
#check
sns.distplot(np.log1p(train['GrLivArea']))
stats.probplot(np.log1p(train['GrLivArea']), plot = plt)

#lets apply the skewness to original colmn
train['SalePrice'] = np.log1p(train['SalePrice'])

ntrain = train.shape[0]
ntest = test.shape[0]

train_id = train['Id']
test_id = test['Id']
train_saleprice = train['SalePrice']

del train['Id']
del test['Id']
del train['SalePrice']

all_data = pd.concat((train,test)).reset_index(drop=True)

#let's check for missing values
info = pd.DataFrame(all_data.dtypes).T
info = info.append(pd.DataFrame(all_data.isnull().sum()).T)
info = info.append(pd.DataFrame(all_data.isnull().sum()/all_data.shape[0] * 100).T)


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


info = pd.DataFrame(all_data.dtypes).T
info = info.append(pd.DataFrame(all_data.isnull().sum()).T)
info = info.append(pd.DataFrame(all_data.isnull().sum()/all_data.shape[0] * 100).T)
# missing data has been replaced

#convert the req columns to categorical(object dtype)
#some columns mentioned as int are actually categorical...so lets convert them to categorical(object--same as str)
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#lets label encode some of the categorical variables
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
    
    
#Now let's separate the categorical and numerical and corrrect the skewness
categorical_features = all_data.select_dtypes(include = ['object']).columns
all_data_categorical = all_data[categorical_features]

numerical_features = all_data.select_dtypes(exclude = ['object']).columns
all_data_numerical = all_data[numerical_features]

data_skewed = all_data_numerical.skew()
data_skewed = data_skewed[abs(data_skewed) > 0.5]
skewed_features = data_skewed.index
all_data_numerical[skewed_features] = np.log1p(all_data_numerical[skewed_features])

all_data = pd.concat([all_data_categorical, all_data_numerical], axis = 1)

#Finally lets convert the categorical to dummy encoding
all_data = pd.get_dummies(all_data)


#Divide the data back into original train and test datasetdataset
orig_train = all_data[:ntrain]
orig_test = all_data[ntrain:]
#train_saleprice is output variable

#divide the training into train and validation
x_train, x_val, y_train, y_val = train_test_split(orig_train, train_saleprice, test_size = 0.30, random_state = 41)

#perform standard scaling
sc = StandardScaler()
x_train.loc[:,numerical_features] = sc.fit_transform(x_train.loc[:,numerical_features])
x_val.loc[:,numerical_features] = sc.transform(x_val.loc[:,numerical_features])


#lets perform cross validation to get better accuracy
#let's start modelling
lr = LinearRegression()
model_lr = lr.fit(x_train, y_train)
y_val_pred_lr = model_lr.predict(x_val)

rmse_lr = sqrt(mean_squared_error(y_val, y_val_pred_lr))


#lets try ridgecv
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

y_val_pred_ridge = ridge.predict(x_val)
rmse_lr_ridge = sqrt(mean_squared_error(y_val, y_val_pred_ridge))
print(rmse_lr_ridge)#0.11429750740258748
#lets visualize the important features
coefs_ridge = pd.Series(ridge.coef_, index=x_train.columns)
imp_coefs = pd.concat([coefs_ridge.sort_values().head(10), coefs_ridge.sort_values().tail(10)]) 
imp_coefs.plot(kind = "barh")
plt.title("imp coeffs of ridge")

print("ridge picked " + str(sum(coefs_ridge != 0)) + " features and eliminated the other " +  \
      str(sum(coefs_ridge == 0)) + " features")


#let's try LassoCV
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1],max_iter = 50000, cv = 10)
lasso.fit(x_train, y_train)
alpha = lasso.alpha_
print("best lasso:", alpha)

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(x_train,y_train)
alpha = lasso.alpha_
print("best alpha:", alpha)

y_val_pred_lasso = lasso.predict(x_val)
rmse_lr_lasso = sqrt(mean_squared_error(y_val, y_val_pred_lasso))
print(rmse_lr_lasso)#0.111861949

#to check the important co-eff after performing lasso
coef = pd.Series(lasso.coef_, index = x_train.columns)
coef.sort_values(ascending = False).head(10)#top 10 features that increase the home price
coef.sort_values().head(10)#top 10 features that decrease the home price

#to visualizr the above features
imp_coefs = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("imp coeffs of lasso")

print("Lasso picked " + str(sum(coef != 0)) + " features and eliminated the other " +  \
      str(sum(coef == 0)) + " features")



#lets try elasticcv
elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(x_train, y_train)
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Try again for more precision with l1_ratio centered around " + str(ratio))
elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(x_train, y_train)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1    
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 
      " and alpha centered around " + str(alpha))
elasticNet = ElasticNetCV(l1_ratio = ratio,
                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 
                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 
                                    alpha * 1.35, alpha * 1.4], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(x_train, y_train)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1    
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

y_val_ela = elasticNet.predict(x_val)
rmse_ela = sqrt(mean_squared_error(y_val, y_val_ela))
print(rmse_ela)#0.1118

# Plot important coefficients
coefs = pd.Series(elasticNet.coef_, index = x_train.columns)
imp_coefs = pd.concat([coefs.sort_values().head(10),coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")
print("elasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")



#let's apply Kernel Ridge
kernalridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
kernalridge.fit(x_train, y_train)
y_val_pred_ker = kernalridge.predict(x_val)

rmse_ker = sqrt(mean_squared_error(y_val, y_val_pred_ker))
print(rmse_ker)#0.1124
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(y_val, y_val_pred_ker)
print(slope, intercept, r_value, p_value, std_err)

plt.scatter(y_val_pred_ker, y_val_pred_ker-y_val, c='g', s=40)


#let's apply decisiontree
dectree = DecisionTreeRegressor()
dectree.fit(x_train, y_train)
y_val_pred_dectree = dectree.predict(x_val)

rmse_dectree = sqrt(mean_squared_error(y_val_pred_dectree, y_val))
print(rmse_dectree)#0.20



#let's apply RandomForest
randfrst = RandomForestRegressor(max_depth = 12)
randfrst.fit(x_train, y_train)
y_val_pred_randfrst = randfrst.predict(x_val)

rmse_randfrst = sqrt(mean_squared_error(y_val_pred_randfrst, y_val))
print(rmse_randfrst)


#ensemble -- many weak learners
#bagging --rand forest -- each bag gets data, repetation is there--count the vote 
#boosting -- AdaBoost -- one bag gets data -- trained and tested on that bag--errors are
#sent to the next bag along with some original random points

#let's apply gradient boosting mechanism
gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
gbr.fit(x_train,y_train)
y_val_gbr = gbr.predict(x_val)

rmse_gbr = sqrt(mean_squared_error(y_val, y_val_gbr))
print(rmse_gbr)#0.118


#let's do using xgboost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)




#modelling in LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(x_train, y_train)
y_val_pred_lgbm = model_lgb.predict(x_val)
rmse_lgb = sqrt(mean_squared_error(y_val, y_val_pred_lgbm))
print(rmse_lgb)#0.1200












