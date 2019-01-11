# -*- coding: utf-8 -*-
# @File  : hptest001.py
# @Author: Panbo
# @Date  : 2019/1/8
# @Desc  : 房价测试V01
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew

#查看数据基本信息
'''
data =pd.read_csv(r'./source data/train.csv')
print(data.head())
print("----------------------------------")
print(format(data.shape))
print("----------------------------------")
print(data.info())
'''
train_data = pd.read_csv(r'./source data/train.csv')
test_data = pd.read_csv(r'./source data/test.csv')
#略过基本信息阶段

#ID这一栏基本没用，干扰信息，考虑是不是删了
train_ID = train_data['Id']
test_ID = test_data['Id']
#train_data.drop("Id", axis= 1, inplace= True)

#数据整理和分析
'''
#研究分析房价数据分布
print(train_data['SalePrice'].describe())
sns.set_style("whitegrid")
sns.distplot(train_data['SalePrice'])


#居住面积和房价
fig, ax =plt.subplots()
ax.scatter( x  = train_data['GrLivArea'], y = train_data['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()

#测试总房间数
data = pd.concat([train_data['SalePrice'], train_data['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')

data = pd.concat([train_data['SalePrice'], train_data['YearBuilt']], axis=1 )
f, ax = plt.subplots()
fig = sns.boxplot(x='YearBuilt', y = 'SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)


fig, ax = plt.subplots()
ax.scatter(x = train_data['GrLivArea'], y = train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
'''



#挑除异常点
'''
train_data = train_data.drop(train_data[train_data['GrLivArea']>4000].index)
fig, ax = plt.subplots()
ax.scatter(train_data['GrLivArea'], train_data['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()

#让数据更加平滑
#log1p就是log（1+x)
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
sns.distplot(train_data['SalePrice'] )

#看一下函数拟合
(mu, sigma) = norm.fit(train_data['SalePrice'])
print('\n mu ={:.2f} and sigma = {:.2f} \n' .format(mu,sigma))

#画个图，看下分布
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#再话个QQ分布，不知道为啥，老外特别喜欢这个，
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
plt.show()
'''


#开始初步的尝试
#1.通用步骤，把test和train联合在一起。
train_data = train_data.drop(train_data[train_data['GrLivArea']>4000].index)
train= train_data.shape[0]
test = test_data.shape[0]
y_train = train_data.SalePrice.values
all_data = pd.concat((train_data, test_data)).reset_index(drop=True,)

all_data.drop(['SalePrice'], axis =1, inplace = True)
#print("总数据的大小为：" .format(all_data.shape))


#查看缺失值
#all_data.info()  #这个相对来说比较简单

#用None补充的数据
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#LotFrontage这个属性，假定为同一个街道的属性相似。

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
#  'GarageType', 'GarageFinish', 'GarageQual' and 'GarageCond' 这几个属性的空缺，十有八九是没有
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
#有一些硬件，没有就是零，然后同一个街道默认为一样
#同时，假定没有车库就没车，（确实不合理，但是好统计）
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

#'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath' 这些没有代表面积为0
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#Utilities这个属性对数据的影响不大，可以将剔除出去
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
#接下来的数据里面，缺失值比较少，用众数来弥补
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#---------------------------------------------------------------------------------------------------
#数据处理结束
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
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
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

#Getting dummy categorical features.
all_data = pd.get_dummies(all_data)
print(all_data.shape)
#Splitting the data back into test and train
train1 = all_data[:train]
test1 = all_data[train:]
train = train1
test  = test1
#-----------------------
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
rfc=RandomForestRegressor(n_estimators=1000)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

rfc.fit(train,y_train)
rfc_train_pred = rfc.predict(train.values)
rfc_pred = np.expm1(rfc.predict(test.values))
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
GBoost.fit(train,y_train)
GBoost_train_pred = GBoost.predict(train)
GB_pred = np.expm1(GBoost.predict(test.values))
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = GB_pred
sub.to_csv('submission.csv',index=False)