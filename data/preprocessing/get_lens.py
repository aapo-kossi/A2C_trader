# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:54:35 2021

@author: Aapo Kössi
"""
import pandas as pd
from numpy import savez as s

df = pd.read_csv('C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader/data/ccm4.csv',
                 usecols = [0,4], parse_dates = True, index_col=[0,1])
startdate = df.index.levels[1].min()
enddate = (df.index.levels[1].max() - startdate).days
groups = df.groupby('GVKEY')
lens = []
for key, df in groups:
    lens.append(df.shape[0])
s('C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader/data/ccm4_raw_metadata',
      lens = lens, startdate = startdate, enddate = enddate)

