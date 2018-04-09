
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', -1)
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# In[53]:


#reading the files
train = pd.read_csv("train-housing.csv")
test = pd.read_csv("test-housing.csv")
print(train.shape)
print(test.shape)
print(train.info())
print(test.info())


# In[54]:


#check for any duplicate rows
print(train['Id'].nunique())
print(train.shape[0])
#no duplicates


# In[55]:


#save the id's
train_id = train['Id']
test_id = test['Id']

del train['Id']
del test['Id']
print(train.shape)
print(test.shape)


# In[56]:


#check for outliers in data
_, bp = pd.DataFrame.boxplot(train['GrLivArea'], return_type='both')

outliers = [flier.get_ydata() for flier in bp["fliers"]]
outliers


# In[57]:


#viewing outliers through scatterplot
#for linear regression, no need to remove all the outliers, remove the ones that are not linear....for this purpose a boxplot
#is not sufficient....hence use scatterplot and remove them
plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()


# In[58]:


#lets remove the outliers that are not linear
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()


# In[59]:


#rather than removing outliers for each variable.....we can use RobustScalar during modeliing


# In[67]:


#Let's analyse the target variable
  #lets check the distribution i.e. is it normal or not
sns.distplot(train['SalePrice'])
plt.show()
print(train['SalePrice'].describe())

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
plt.show()


# In[68]:


#the distribution seems to be right skewed, hence let's apply some transfrmatins

#let's check if the distribution is normal...when we apply np.log1p
sns.distplot(np.log1p(train['SalePrice']))
plt.show()

fig = plt.figure()
stats.probplot(np.log1p(train['SalePrice']), plot = plt)
plt.show()


# In[69]:


#the data seems to have become normal..hence replace the orig colm with log(orig colm)
train['SalePrice'] = np.log1p(train['SalePrice'])

#just save the SalePrice, just in case
train_final_y = train['SalePrice']


# In[78]:


#let's check the correlation
plt.figure(figsize = (16,16))
sns.heatmap(train.corr(),linewidths=.5, fmt="d")
plt.show()


# In[71]:


#Now let's do feature engineering, imputing missing values.....
# let's concatenate the train and test values
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train,test)).reset_index(drop=True)
del all_data['SalePrice']
print(all_data.shape)


# In[88]:


#lets start imputing missing values
all_data_info=pd.DataFrame(all_data.dtypes).T.rename(index={0:'column type'})
all_data_info=all_data_info.append(pd.DataFrame(all_data.isnull().sum()).T.rename(index={0:'null values (nb)'}))
all_data_info=all_data_info.append(pd.DataFrame(all_data.isnull().sum()/all_data.shape[0]*100).T.rename(index={0:'null values (%)'}))
all_data_info


# In[97]:


#lets start imputing missing values
all_data_info=pd.DataFrame(all_data.dtypes).T
all_data_info=all_data_info.append(pd.DataFrame(all_data.isnull().sum()).T)
all_data_info=all_data_info.append(pd.DataFrame(all_data.isnull().sum()/all_data.shape[0]*100).T)
all_data_info


# In[101]:


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


# In[102]:


all_data_info=pd.DataFrame(all_data.dtypes).T
all_data_info=all_data_info.append(pd.DataFrame(all_data.isnull().sum()).T)
all_data_info=all_data_info.append(pd.DataFrame(all_data.isnull().sum()/all_data.shape[0]*100).T)
all_data_info


# In[103]:


all_data.info()


# In[104]:


all_data.head(5)


# In[107]:


#some columns mentioned as int are actually categorical...so lets convert them to categorical type
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# In[108]:


all_data.info()


# In[109]:


pd.value_counts(all_data['BsmtCond'])


# In[110]:


all_data.head(10)


# In[111]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# In[112]:


all_data.head(10)


# In[113]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[114]:


all_data.shape


# In[115]:


#lets check the skewness of all numerical variables and try to makwe them normal and also reduce the effect of outliers
#divide the data into categorical and numerical features
categorical_features = all_data.select_dtypes(include = ['object']).columns
all_data_categorical = all_data[categorical_features]

numerical_features = all_data.select_dtypes(exclude = ['object']).columns
all_data_numerical = all_data[numerical_features]
print(all_data_categorical.shape)
print(all_data_numerical.shape)


# In[157]:


#apply skewness to all the numerical columns and transform the positive sjewed using nplog1p transform
#calculating skewness of each colm in df
skewness_val = all_data_numerical.skew()
skewness_val


# In[158]:


skewness_val = skewness_val[abs(skewness_val)>0.5]
skewness_val


# In[159]:


skewed_features = skewness_val.index
skewed_features


# In[160]:


all_data_numerical[skewed_features] = np.log1p(all_data_numerical[skewed_features])


# In[164]:


#lets combine all the values back into one datset
all_data_done = pd.concat([all_data_categorical, all_data_numerical], axis=1)
print(all_data_done.shape)


# In[165]:


#Finally lets convert the categorical to dummy encoding
all_data_done = pd.get_dummies(all_data_done)
print(all_data_done.shape)


# In[ ]:


####################


# In[172]:


#Divide the data back into original dataset
train_final_x = all_data_done[:ntrain]
test_final_x = all_data_done[ntrain:]
print(train_final_x.shape)#2 observations were deleted in train because of outliers
print(test_final_x.shape)
print(train_final_y.shape)


# In[178]:


#lets standardize the features
#Standardization cannot be done before the partitioning, as we don't want to fit the StandardScaler on some 
#observations that will later be used in the test set.
sc = StandardScaler()
train_final_x.loc[:, numerical_features] = sc.fit_transform(train_final_x.loc[:, numerical_features])
test_final_x.loc[:, numerical_features] = sc.transform(test_final_x.loc[:, numerical_features])

