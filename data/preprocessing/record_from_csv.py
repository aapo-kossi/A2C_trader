# -*- coding: utf-8 -*-
"""
Created on Mon May 17 08:32:35 2021

@author: Aapo Kössi
"""

import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import datetime
sys.path.append('C:/users/aapok/AerialRealTimeVehicleRecognition/TensorFlow/workspace/trader/A2C_trader')
import constants

def onehot_encode(lst):
    as_df = pd.DataFrame({'categories' : lst})
    one_hot = pd.get_dummies(as_df['categories'], dtype=np.bool_)
    return one_hot


start = datetime.date(1990,1,1)
end = datetime.date(2021, 3, 31)

parser = argparse.ArgumentParser(description='Train a neural network to trade n stocks concurrently, '\
                                             'provided a .csv file of stock data.')

parser.add_argument('-f', '--filepath', help = 'path of the .csv input file', type=str)

arg = parser.parse_args()
filepath = arg.filepath

df = pd.read_csv(filepath, index_col = [0,2])


print('got the dataframe')
df = df[df['curcdd']=='USD']

print(df.head())
df[['prccd','prchd','prcld','prcod','divd','divsp']] = \
    df[['prccd','prchd','prcld','prcod','divd','divsp']].multiply(
    df['ajexdi'] ** -1,axis = 'index')
df = df.fillna(0)
df['dist'] = df['cheqv'] + df['divd'] + df['divsp']
df.drop(['ajexdi','curcdd','iid','cheqv','divd','divsp'],axis='columns', inplace=True)
print('did initial preprocessing')
ticker_dfs = list(df.groupby('GVKEY', sort=False))
nested_datalists= []
sector_list = []
industry_list = []
lens = []
key_list = []
conames = []
print('starting loop')
for gvkey, df in ticker_dfs:
    if df.shape[0] < constants.MIN_DAYS_AVLB: continue
    # print('enough datapoints to include')
    df.sort_index(inplace=True)
    sector_list.append(str(int(df.gsector.iloc[0])))
    industry_list.append(str(int(df.gind.iloc[0])))
    conames.append(df['conm'].iloc[0])
    df.drop(['conm'],axis=1,inplace = True)
    unique = ~df.index.duplicated(keep='first')
    df = df[unique]
    df = df.reset_index(level=1)
    df.rename({'datadate':'date'},axis=1,inplace=True)
    df['date'] = df['date'].astype('float')
    # print('dropped duplicate days')
    nested_datalists.append(df)
    lens.append(df.shape[0])
    key_list.append(gvkey)
print('finished loop')
print(f'longest single stock dataset: {max(lens)}')
raise SystemExit
# df = pd.concat(processed_dfs).reset_index(level = 1)
print('got complete nested data')
complete = pd.concat(nested_datalists)
dataset = tf.data.Dataset.from_tensor_slices((complete,))
data_index = df.columns
ticker_dict = dict(zip(conames,key_list))
onehot_sectors = onehot_encode(sector_list)
onehot_industries = onehot_encode(industry_list)
onehot_cats = onehot_sectors.join(onehot_industries,rsuffix='_ind')
num_tickers = len(conames)
# complete_data = FileData(dataset, data_index, ticker_dict, onehot_cats, num_tickers, lens)
# print(complete_data.dataset.element_spec)
# Records.write_record(complete_data, file_name='CCM1v2')


