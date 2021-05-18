# -*- coding: utf-8 -*-
"""
Created on Mon May 17 08:32:35 2021

@author: Aapo Kössi
"""

import os
import pandas as pd
import tensorflow as tf
import argparse
import datetime

start = datetime.date(1990,1,1)
end = datetime.date(2021, 3, 31)
dti = pd.date_range(start=start,end=end)

parser = argparse.ArgumentParser(description='generate tfrecord file for model training, '\
                                             'provided a .csv file of stock data.')

parser.add_argument('-f', '--filepath', help = 'path of the .csv input file', type=str)

arg = parser.parse_args()
filepath = arg.filepath

df = pd.read_csv(filepath, index_col = [0,2])
df = df.index.set_levels(pd.to_datetime(df.index.get_level_values(1)),level=1)
df = df[df['curcdd']=='USD']

df[['prccd','prchd','prcld','prcod','divd','divsp']] *= df['ajexdi'] ** -1
df['dist'] = df['cheqv'] + df['divd'] + df['divsp']
df = df.drop(['ajexdi','curcdd','iid','cheqv','divd','divsp'],axis='columns')
ticker_dfs = list(df.groupby('GVKEY', sort=False))
processed_dfs = []
for gvkey, df in ticker_dfs:
    df = df.sort_index()
    df = df.reindex(dti, fill_value = 0)
    processed_dfs.append(df)
sparse_df = pd.concat(processed_dfs)
