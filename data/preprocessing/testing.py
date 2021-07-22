# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:54:35 2021

@author: Aapo Kössi
"""
import sys
import pandas as pd
sys.path.append('C:/users/aapok/AerialRealTimeVehicleRecognition/TensorFlow/workspace/trader/A2C_trader')

df = pd.read_csv('C:/Users/aapok/AerialRealTimeVehicleRecognition/TensorFlow/workspace/trader/A2C_trader/data/ccm_1.csv',
                 index_col = [0,2])
unique = ~df.index.duplicated(keep='first')
df = df[unique]
lens = df.groupby('GVKEY').count()['prccd']
print(max(lens))
