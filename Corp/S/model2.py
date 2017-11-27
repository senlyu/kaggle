# coding: utf-8




import pandas as pd
from datetime import timedelta
import datetime

#---------------------------------------------------------
print('start',datetime.datetime.now())

dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32'}

train = pd.read_csv('~/git/data/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'],
                    skiprows=range(1, 86672217) #Skip dates before 2016-08-01
                    #nrows=1000000
                    )


# In[20]:

#---------------------------------------------------------
print('finish loading',datetime.datetime.now())

train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
train['dow'] = train['date'].dt.dayofweek


# In[21]:

# creating records for all items, in all markets on all dates
# for correct calculation of daily unit sales averages.
#---------------------------------------------------------
print('finish data',datetime.datetime.now())
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()
train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
)


# In[22]:

#---------------------------------------------------------
print('finish sep',datetime.datetime.now())
del u_dates, u_stores, u_items

train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs
train.reset_index(inplace=True) # reset index and restoring unique columns  
lastdate = train.iloc[train.shape[0]-1].date


# In[26]:

#---------------------------------------------------------
print('get lastdate',datetime.datetime.now())
#Days of Week Means
#By tarobxl: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/42948
ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw')
ma_dw.reset_index(inplace=True)
ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk')
ma_wk.reset_index(inplace=True)




# In[ ]:

#-------------------------------------------
# normalization
se_max = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].quantile(0.9,interpolation='lower').to_frame('semax')
se_max.reset_index(inplace=True)

se_min = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].quantile(0.1,interpolation='lower').to_frame('semin')
se_min.reset_index(inplace=True)

train = pd.merge(train, se_max, how='left', on=['item_nbr','store_nbr','dow'])
train = pd.merge(train, se_min, how='left', on=['item_nbr','store_nbr','dow'])
train['unit_sales'] = (train['unit_sales'] - train['semin'])/(train['semax'] - train['semin'])
train.drop('semax', axis=1, inplace=True)
train.drop('semin', axis=1, inplace=True)

train.loc[(train.unit_sales<0),'unit_sales'] = 0
train.loc[(train.unit_sales>1),'unit_sales'] = 1



#---------------------------------------------------------
print('get ma_dw ma_wk',datetime.datetime.now())
#Moving Averages
ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais226')
for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')


# In[ ]:

#---------------------------------------------------------
print('get MA',datetime.datetime.now())

se_max.to_csv('model2_se_max.csv')
se_min.to_csv('model2_se_min.csv')

del tmp,tmpg,train

ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)


# In[ ]:

#---------------------------------------------------------
print('get median of MA',datetime.datetime.now())
#Load test
test = pd.read_csv('~/git/data/test.csv', dtype=dtypes, parse_dates=['date'])
test['dow'] = test['date'].dt.dayofweek
test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])
test = pd.merge(test, se_max, how='left', on=['item_nbr','store_nbr','dow'])
test = pd.merge(test, se_min, how='left', on=['item_nbr','store_nbr','dow'])
test.to_csv('model2_test.csv')

# In[ ]:

#---------------------------------------------------------
print('get test',datetime.datetime.now())
del ma_is, ma_wk, ma_dw, se_max, se_min

#Forecasting Test
test['unit_sales'] = test.mais 
pos_idx = test['mawk'] > 0
test_pos = test.loc[pos_idx]
test.loc[pos_idx, 'unit_sales'] = test_pos['mais'] * test_pos['madw'] / test_pos['mawk']

test['unit_sales'] = test['unit_sales'] * (test['semax'] - test['semin']) + test['semin']

test.loc[:, "unit_sales"].fillna(0, inplace=True)
test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1) # restoring unit values 


# In[ ]:

#---------------------------------------------------------
print('get results',datetime.datetime.now())
#50% more for promotion items
test.loc[test['onpromotion'] == True, 'unit_sales'] *= 1.5

test[['id','unit_sales']].to_csv('model2.csv.gz', index=False, float_format='%.3f', compression='gzip')

print('nulls', test[test['unit_sales']<0])
# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



