
# coding: utf-8

# In[ ]:

import pandas as pd
from datetime import timedelta
import datetime
#---------------------------------------------------------
print('start',datetime.datetime.now())


# In[ ]:

dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32'}

train = pd.read_csv('~/git/data/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'],
                    skiprows=range(1, 86672217) #Skip dates before 2016-08-01
                    #nrows=60000
                    )
#---------------------------------------------------------
print('finish loading',datetime.datetime.now())


# In[ ]:

train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
train['dow'] = train['date'].dt.dayofweek
train['month'] = train['date'].dt.month
# creating records for all items, in all markets on all dates
# for correct calculation of daily unit sales averages.
#---------------------------------------------------------
print('finish data',datetime.datetime.now())


# In[ ]:

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
#---------------------------------------------------------
print('finish sep',datetime.datetime.now())


# In[ ]:

del u_dates, u_stores, u_items

train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs
train.reset_index(inplace=True) # reset index and restoring unique columns  
lastdate = train.iloc[train.shape[0]-1].date
#---------------------------------------------------------
print('get lastdate',datetime.datetime.now())


# In[ ]:

#Days of Week Means
#By tarobxl: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/42948
ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw')
ma_dw.reset_index(inplace=True)
ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk')
ma_wk.reset_index(inplace=True)

ma_mn = train[['item_nbr','store_nbr','month','unit_sales']].groupby(['item_nbr','store_nbr','month'])['unit_sales'].mean().to_frame('mamn')
ma_mn.reset_index(inplace=True)
ma_me = ma_mn[['item_nbr','store_nbr','mamn']].groupby(['store_nbr', 'item_nbr'])['mamn'].mean().to_frame('mame')
ma_me.reset_index(inplace=True)
#---------------------------------------------------------
print('get ma_dw ma_wk',datetime.datetime.now())


# In[ ]:

#Moving Averages
ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais226')
for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')
#---------------------------------------------------------
print('get MA',datetime.datetime.now())


# In[ ]:

del tmp,tmpg,train

ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)
#---------------------------------------------------------
print('get median of MA',datetime.datetime.now())


# In[ ]:

#Load test
test = pd.read_csv('~/git/data/test.csv', dtype=dtypes, parse_dates=['date'])
test['dow'] = test['date'].dt.dayofweek
test['month'] = test['date'].dt.month
test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_me, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])
test = pd.merge(test, ma_mn, how='left', on=['item_nbr','store_nbr','month'])
#---------------------------------------------------------
print('get test',datetime.datetime.now())
test.to_csv('model3_test.csv')

# In[ ]:

del ma_is, ma_wk, ma_dw, ma_mn, ma_me

#Forecasting Test
test['unit_sales'] = test.mais 

pos_idx = test['mawk'] > 0
test_pos = test.loc[pos_idx]
test.loc[pos_idx, 'unit_sales'] = test_pos['unit_sales'] * test_pos['madw'] / test_pos['mawk']

pos_idx = test['mame'] > 0
test_pos = test.loc[pos_idx]
test.loc[pos_idx, 'unit_sales'] = test_pos['unit_sales'] * test_pos['mamn'] / test_pos['mame']

test.loc[:, "unit_sales"].fillna(0, inplace=True)
test['unit_sales'] = test['unit_sales'].apply(pd.np.expm1) # restoring unit values 
#---------------------------------------------------------
print('get results',datetime.datetime.now())


# In[ ]:

#50% more for promotion items
test.loc[test['onpromotion'] == True, 'unit_sales'] *= 1.5

test[['id','unit_sales']].to_csv('model3.csv.gz', index=False, float_format='%.3f', compression='gzip')


# In[ ]:




# In[ ]:



