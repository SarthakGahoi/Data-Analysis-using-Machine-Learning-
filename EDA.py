import random
random.seed(2024)


# importing libraries

import missingno as msno
import numpy as np
from scipy.stats import shapiro
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# importing from sklearn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import load_diabetes , load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics  import root_mean_squared_error
from  sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from fasteda import fast_eda


# hiding warnings

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn =warn
# warnings.filterwarnings('ignore')

# REGRESSION MODEL

# load dataset

diabetes_X , diabetes_y =load_diabetes(return_X_y= True , as_frame=True , scaled=False)
diabetes= pd.concat([diabetes_X,pd.Series(diabetes_y)],axis=1). rename({0:'target'},axis=1)
diabetes_default = load_diabetes()
# print(diabetes_default['DESCR'])
(diabetes.head(5))
(diabetes.isna().max(axis=0).max())
missing_cols=random.sample(range(len(diabetes.columns)-1),3)
missing_rows=random.sample(diabetes.index.tolist(), int(np.round(len(diabetes.index.tolist())/10)))
diabetes.iloc[missing_rows,missing_cols]=np.nan

enc1=OneHotEncoder(handle_unknown='ignore' , drop=None)
encoded_sex=enc1.fit_transform(diabetes[['sex']]).toarray()
encoded_sex=pd.DataFrame(encoded_sex,columns=['sex'+str(int(x))for x in enc1.categories_[0]])
diabetes=pd.concat([diabetes,encoded_sex],axis=1)
diabetes=diabetes.drop(['sex','sex2'],axis=1)

cols = list(diabetes.columns)
cols.insert(1, cols.pop(cols.index('sex1')))
diabetes = diabetes[cols]
x_train,x_test,y_train,y_test=train_test_split(diabetes.iloc[ : , : -1],diabetes.iloc[:,[-1]],test_size=0.33,random_state=2004)
msno.matrix(diabetes)
var = diabetes.index[diabetes['bmi'].isna()]

# linear regression droping nan

nonnan_train_indices=x_train.index[~x_train.isna().max(axis=1)]
nonnan_test_indices=x_test.index[~x_test.isna().max(axis=1)]
reg=LinearRegression().fit(x_train.loc[nonnan_train_indices],y_train.loc[nonnan_train_indices])
pred=reg.predict(x_test.loc[nonnan_test_indices])
root_mean_squared_error(y_test.loc[nonnan_test_indices],pred)

# linear regression with mean fill

nonnann_train_indices=x_train.loc[~x_train.isna().max(axis=1)]
nonnann_test_indices=x_test.loc[~x_test.isna().max(axis=1)]
imp_mean=SimpleImputer(missing_values=np.nan,strategy ='mean')
imp_mean.fit(x_train)
x_train_meanfilled=imp_mean.transform(x_train)
reg=LinearRegression().fit(x_train_meanfilled,y_train)
pred = reg.predict(x_test.loc[nonnan_test_indices])
root_mean_squared_error(y_test.loc[nonnan_test_indices],pred)

#linear regression with median fill

nonnan_train_indices=x_train.index[~x_train.isna().max(axis=1)]
nonnan_test_indices=x_test.index[~x_test.isna().max(axis=1)]
imp_median=SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(x_train)
x_train_median_filled=imp_mean.transform(x_train)
reg=LinearRegression().fit(x_train_median_filled,y_train)
pred=reg.predict(x_test.loc[nonnan_test_indices])
root_mean_squared_error(y_test.loc[nonnan_test_indices],pred)

# Histogram and Box_plot


for idx , col in enumerate (['s3']):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    sns.histplot(diabetes,x =diabetes[col],kde=True,
                 color=sns.color_palette("hls",len(['s3']))[idx],ax=ax1)
    sns.boxplot(x=diabetes[col],width=0.4,linewidth=3,fliersize=2.5,
                color =sns.color_palette("hls",len(['s3']))[idx],ax=ax2)
    fig.suptitle(f"Histogram and Boxplot of {col}",size=20,y=1.0)
    plt.show()

#normality test on s3
stat , p =shapiro(np.log(x_train['s3']))
print('Statistics =%.3f,p=%.3f'%(stat,p))
alpha=0.5
if p > alpha :
    print('Sample looks normally distributed (fail to reject H0)')
else:
    print("Sample does not look normally distributed (reject H0)")

x_train['s3']=np.log(x_train['s3'])
x_test['s3']=np.log(x_test['s3'])

nonnan_train_indices = x_train.index[~x_train.isna().max(axis=1)]
nonnan_test_indices=x_test.index[~x_test.isna().max(axis=1)]

imp_median=SimpleImputer(missing_values=np.nan,strategy='median')
imp_median.fit(x_train)

x_train_median_filled=imp_median.transform(x_train)
reg=LinearRegression().fit(x_train_median_filled,y_train)

pred=reg.predict(x_test.loc[nonnan_test_indices])

root_mean_squared_error(y_test.loc[nonnan_test_indices],pred)

for idx, col in enumerate (['s2']):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
    sns.histplot(diabetes,x=diabetes[col],kde=True,
                 color=sns.color_palette("hls",len(['s2']))[idx],ax=ax1)
    sns.boxplot(x=diabetes[col],width=0.4,linewidth=3,fliersize=2.5,
                color=sns.color_palette("hls",len(['s2']))[idx],ax=ax2)
    fig.suptitle(f"Histogram and Boxplot of {col}",size=20,y = 0.8)
    plt.show()

#outlier

x_train_nonoutlier_idx=x_train.index[x_train.s2<x_train.s2.quantile(0.999)]
x_train=x_train.loc[x_train_nonoutlier_idx]
y_train=y_train.loc[x_train_nonoutlier_idx]

nonnan_train_indices=x_train.index[~x_train.isna().max(axis=1)]
nonnan_test_indices= x_test.index[~x_test.isna().max(axis=1)]

imp_median= SimpleImputer(missing_values=np.nan,strategy='median')
imp_median.fit(x_train)
x_train_median_filled=imp_median.transform(x_train)
reg=LinearRegression().fit(x_train_median_filled,y_train)
pred=reg.predict(x_test.loc[nonnan_test_indices])
root_mean_squared_error(y_test.loc[nonnan_test_indices],pred)

#correlation matrix

plt.figure(figsize=(12,8))
sns.heatmap(diabetes.corr(),annot=True,cmap="Spectral",linewidths=2,linecolor="#000000",fmt='.3f')
plt.show()

from IPython.display import display  # ✅ Add this

cols_no_s1 = [i for i in x_train.columns if i != 's1']
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(x_train.loc[:, cols_no_s1])
x_train_median_filled = imp_median.transform(x_train.loc[:, cols_no_s1])

reg = LinearRegression().fit(x_train_median_filled, y_train)
pred = reg.predict(x_test.loc[nonnan_test_indices, cols_no_s1])
root_mean_squared_error(y_test.loc[nonnan_test_indices], pred)

sns.pairplot(diabetes)
plt.show()

fast_eda(diabetes)  # Now this will work correctly



# classification

iris_sklearn=load_iris()
iris_npy=np.concatenate([iris_sklearn['data'],np.atleast_2d(iris_sklearn['target']).T],axis=1)
col_names=['sepal_length','sepal_width','petal_length','petal_width','target']
iris=pd.DataFrame(iris_npy,columns=col_names)
print(iris_sklearn['DESCR'])
iris['target'].sample(5)

class_names=dict(zip(list(map(float,range(len(iris_sklearn['target_names'])))),iris_sklearn['target_names']))

# EDA with iris dataset

fast_eda(iris,target='target')
plt.axis('equal')
sns.scatterplot(iris,x='petal_width',y='sepal_width',hue='target',palette=sns.color_palette("hls",iris['target'].nunique()))
plt.show()

for idx,col in enumerate (['sepal_length']):
    fig,(ax1,ax2)=plt.subplots(1,2,figszie=(14,6))
    sns.histplot(iris,x=iris[col],kde=True,
                 color=sns.color_palette("hls",iris['target'].nunique()),ax=ax1,hue="target")
    sns.boxplot(x=iris[col],width=0.4,linewidth=3,fliersize=2.5,
                color=sns.color_palette("hls",iris['target'].nunique())[idx],ax=ax2)
    fig.suptitle(f"Histogram and Boxplot of {col}",size=20,y=0.7)
    plt.show()

vc=iris['target'].value_counts()
_=plt.pie(vc)

def autopct_format(values):
    def my_format(pct):
        total=sum(values)
        val=int(round(pct*total/100.0))
        return'{:.1f}%\n({v:d}'.format(pct,v=val)
    return my_format
vc=iris['target'].value_counts()
_=plt.pie(vc,labels=vc.rename(class_names).index,autopct=autopct_format(vc))
