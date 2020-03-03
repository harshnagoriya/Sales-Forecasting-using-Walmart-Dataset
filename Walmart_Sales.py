#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pylab as plt



# In[2]:


train = pd.read_csv('C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Walmart Sales\\input\\train.csv')
feature = pd.read_csv('C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Walmart Sales\\input\\features.csv')
test = pd.read_csv('C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Walmart Sales\\input\\test.csv')
stores = pd.read_csv('C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Walmart Sales\\input\\stores.csv')


# In[3]:


import xlsxwriter
writer=pd.ExcelWriter('C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Walmart Sales\\output\\output.xlsx', engine='xlsxwriter')


# In[ ]:





# In[ ]:





# In[4]:


train_bt = pd.merge(train,stores) 
train = pd.merge(train_bt,feature)
test_bt = pd.merge(test,stores)
test= pd.merge(test_bt,feature)


# In[5]:


train.head(2)


# In[6]:


test.head(2)


# In[ ]:





# In[7]:


print(train.info())


# In[8]:


numvartrain=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
catvartrain=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object']]


# In[9]:


train_num=train[numvartrain]
train_cat=train[catvartrain]
print(numvartrain)
print(catvartrain)


# In[10]:


def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])


# In[11]:


num_summary=train_num.apply(lambda x: var_summary(x)).T
num_summary.to_excel(writer,'Numeric_variable Summary',index=True)
num_summary


# In[12]:


def cat_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.value_counts()], 
                  index=['N', 'NMISS', 'ColumnsNames'])

cat_summary=train_cat.apply(lambda x: cat_summary(x))
cat_summary


# In[13]:


numvartest=[key for key in dict(test.dtypes) if dict(test.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
catvartest=[key for key in dict(test.dtypes) if dict(test.dtypes)[key] in ['object']]


# In[14]:


test_num=test[numvartest]
test_cat=test[catvartest]
print(numvartest)
print(catvartest)


# In[15]:


num_summary=test_num.apply(lambda x: var_summary(x)).T
num_summary.head()


# In[16]:


def cat_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.value_counts()], 
                  index=['N', 'NMISS', 'ColumnsNames'])

cat_summary=test_cat.apply(lambda x: cat_summary(x))
cat_summary


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


train_corr=pd.DataFrame(train.corr())
train_corr.to_excel(writer,'Train_Data Corr',index=True)
train_corr.head()


# In[18]:


test_corr=pd.DataFrame(test.corr())
test_corr.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


print(train.isnull().sum())
print("*"*30)
print(test.isnull().sum())


# In[20]:


test['CPI']=test.groupby(['Dept'])['CPI'].transform(lambda x: x.fillna(x.mean()))
test['Unemployment']=test.groupby(['Dept'])['Unemployment'].transform(lambda x: x.fillna(x.mean()))


# In[21]:


train=train.fillna(0)
test=test.fillna(0)


# In[22]:


print(train.isnull().sum())
print("*"*30)
print(test.isnull().sum())


# In[23]:


train.Weekly_Sales=np.where(train.Weekly_Sales>100000, 100000,train.Weekly_Sales)


# In[ ]:





# In[24]:


train.info()


# In[25]:


train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])


# In[26]:


train['Date_dayofweek'] =train['Date'].dt.dayofweek
train['Date_month'] =train['Date'].dt.month 
train['Date_year'] =train['Date'].dt.year
train['Date_day'] =train['Date'].dt.day 

test['Date_dayofweek'] =test['Date'].dt.dayofweek
test['Date_month'] =test['Date'].dt.month 
test['Date_year'] =test['Date'].dt.year
test['Date_day'] =test['Date'].dt.day


# In[27]:


print (train.Type.value_counts())
print ("*"*30)
print (test.Type.value_counts())


# In[28]:


print(train.IsHoliday.value_counts())
print("*"*30)
print(test.IsHoliday.value_counts())


# In[29]:


train_test_data = [train, test]


# In[30]:


type_mapping = {"A": 1, "B": 2, "C": 3}
for dataset in train_test_data:
    dataset['Type'] = dataset['Type'].map(type_mapping)


# In[31]:


type_mapping = {False: 0, True: 1}
for dataset in train_test_data:
    dataset['IsHoliday'] = dataset['IsHoliday'].map(type_mapping)


# In[32]:


import datetime


# In[33]:


train['Super_Bowl'] = (np.where((train['Date']==datetime.date(2010, 2, 12)) | (train['Date']==datetime.date(2011, 2, 11)) | (train['Date']==datetime.date(2012, 2, 10)) | (train['Date']==datetime.date(2013, 2, 8)),1,0))
train['Labour_Day'] = (np.where((train['Date']==datetime.date(2010, 9, 10)) | (train['Date']==datetime.date(2011, 9, 9)) | (train['Date']==datetime.date(2012, 9, 7)) | (train['Date']==datetime.date(2013, 9, 6)),1,0))
train['Thanksgiving'] = (np.where((train['Date']==datetime.date(2010, 11, 26)) | (train['Date']==datetime.date(2011, 11, 25)) | (train['Date']==datetime.date(2012, 11, 23)) | (train['Date']==datetime.date(2013, 11, 29)),1,0))
train['Christmas'] = (np.where((train['Date']==datetime.date(2010, 12, 31)) | (train['Date']==datetime.date(2011, 12, 30)) | (train['Date']==datetime.date(2012, 12, 28)) | (train['Date']==datetime.date(2013, 12, 27)),1,0))
test['Super_Bowl'] = (np.where((test['Date']==datetime.date(2010, 2, 12)) | (test['Date']==datetime.date(2011, 2, 11)) | (test['Date']==datetime.date(2012, 2, 10)) | (test['Date']==datetime.date(2013, 2, 8)),1,0))
test['Labour_Day'] = (np.where((test['Date']==datetime.date(2010, 9, 10)) | (test['Date']==datetime.date(2011, 9, 9)) | (test['Date']==datetime.date(2012, 9, 7)) | (test['Date']==datetime.date(2013, 9, 6)),1,0))
test['Thanksgiving'] = (np.where((test['Date']==datetime.date(2010, 11, 26)) | (test['Date']==datetime.date(2011, 11, 25)) | (test['Date']==datetime.date(2012, 11, 23)) | (test['Date']==datetime.date(2013, 11, 29)),1,0))
test['Christmas'] = (np.where((test['Date']==datetime.date(2010, 12, 31)) | (test['Date']==datetime.date(2011, 12, 30)) | (test['Date']==datetime.date(2012, 12, 28)) | (test['Date']==datetime.date(2013, 12, 27)),1,0))


# In[34]:


train['IsHoliday']=train['IsHoliday']|train['Super_Bowl']|train['Labour_Day']|train['Thanksgiving']|train['Christmas']
test['IsHoliday']=test['IsHoliday']|test['Super_Bowl']|test['Labour_Day']|test['Thanksgiving']|test['Christmas']


# In[35]:


print (train.Christmas.value_counts())
print (train.Super_Bowl.value_counts())
print (train.Thanksgiving.value_counts())
print (train.Labour_Day.value_counts())


# In[36]:


print (test.Christmas.value_counts())
print (test.Super_Bowl.value_counts())
print (test.Thanksgiving.value_counts())
print (test.Labour_Day.value_counts())


# In[37]:


dp=['Super_Bowl','Labour_Day','Thanksgiving','Christmas']
train.drop(dp,axis=1,inplace=True)
test.drop(dp,axis=1,inplace=True)


# In[38]:


train.info()


# In[39]:


features_drop=['Unemployment','CPI','MarkDown5']
train=train.drop(features_drop, axis=1)
test=test.drop(features_drop, axis=1)


# In[40]:



train.head(2)


# In[41]:


test.head(2)


# In[42]:


train_X=train.drop(['Weekly_Sales','Date'], axis=1)
train_y=train['Weekly_Sales'] 
test_X=test.drop('Date',axis=1).copy()

train_X.shape, train_y.shape, test_X.shape


# In[43]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(train_X, train_y)
y_pred_linear=clf.predict(test_X)
acc_linear=round( clf.score(train_X, train_y) * 100, 2)
print ('score:'+str(acc_linear) + ' percent')


# In[ ]:





# In[ ]:





# In[44]:


from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()
clf.fit(train_X, train_y)
y_pred_dt= clf.predict(test_X)
acc_dt = round( clf.score(train_X, train_y) * 100, 2)
print (str(acc_dt) + ' percent')


# In[45]:


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=10)
clf.fit(train_X, train_y)
y_pred_rf=clf.predict(test_X)
acc_rf= round(clf.score(train_X, train_y) * 100, 2)
print ("Accuracy: %i %% \n"%acc_rf)


# In[ ]:





# In[ ]:





# In[46]:


models = pd.DataFrame({
    'Model': ['Linear Regression','Random Forest','Decision Tree'],
    
    'Score': [acc_linear, acc_rf,acc_dt]
    })

models.sort_values(by='Score', ascending=False)


# In[49]:


submission = pd.DataFrame({
        "Store_Dept_Date": test.Store.astype(str)+'_'+test.Dept.astype(str)+'_'+test.Date.astype(str),
        "Weekly_Sales": y_pred_rf
    })

submission.to_csv('C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Walmart Sales\\output\\OP_sales_pred.csv', index=False)
submission.to_excel(writer,'C:\\Users\\harsh\\Desktop\\HSNProjects\\ML\\Walmart Sales\\output\\OP_sales_Pred',index=False)


# In[ ]:


submission.head()


# In[ ]:




