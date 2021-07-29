# -*- coding: utf-8 -*-
"""
Created on Mon May 17 08:32:35 2021

@author: Aapo Kössi
"""

import sys
import time
import pandas as pd
import numpy as np
import argparse
sys.path.append('C:/users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader')
import constants
import colorama

def get_cats(lst):
    unique = pd.Series(lst).unique()[:-1]
    return unique

def datetime_to_number(dt_index, zero):
    first = zero
    numerical_index = dt_index.map(lambda val: (val - first).days.astype('float'))
    return numerical_index

def onehot_encode(lst):
    as_df = pd.DataFrame({'categories' : lst})
    one_hot = pd.get_dummies(as_df['categories'], dtype=np.bool_)
    return one_hot

def preprocess_chunk(df, dates, arrs, min_days):
    startdate, (start, end) = dates
    df = df[df['curcdd'] == 'USD']
    unique = ~df.index.duplicated(keep='first')
    df = df[unique]
    df = df.loc[df.index.get_level_values(1) < end]
    df = df.loc[df.index.get_level_values(1) >= start]
    if df.shape[0] < min_days: return None, arrs
    avg_vol = df['cshtrd'].sum() / df.shape[0]
    if avg_vol < constants.MIN_AVG_VOL: return None, arrs
    # print('enough datapoints to include')
    df.sort_index(inplace=True)
    df['ajexdi'] = df['ajexdi'].replace(0,1)
    df[['prccd','prchd','prcld','prcod','divd','divsp']] = \
        df[['prccd','prchd','prcld','prcod','divd','divsp']].multiply(
        df['ajexdi'] ** -1,axis = 'index')
    df.fillna(0, inplace=True)
    df['dist'] = df['cheqv'] + df['divd'] + df['divsp']
    df.drop(['ajexdi','cheqv','divd','divsp', 'curcdd'],axis='columns', inplace=True)

    df.reset_index(level=1, inplace=True)
    df.rename({'datadate':'date'},axis=1,inplace=True)
    # df['date'] = df['date'].astype('float')
    # print('dropped duplicate days')
    tgtmap = {'lens': df.shape[0], 'conames': df.conm.iloc[0], 'sector_list': df.gsector.iloc[0]}
    for key in ['lens', 'conames', 'sector_list']:
        try:
            arr = arrs[key]
            arrs[key] = np.append(arr, tgtmap[key])
        except KeyError: continue
    df.drop(['conm'],axis=1,inplace = True)
    return df, arrs




def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    colorama.init()
    print('started')
    


    parser = argparse.ArgumentParser(description='2nd part of preprocessing wrds data after getting batch lengths')

    parser.add_argument('-f', '--filepath', help = 'path of the .csv input file', type=str)
    args = parser.parse_args()

    filepath = args.filepath

    read = 0
    raw_arrs = np.load(filepath + '_raw_lens.npz', allow_pickle = True)
    startdate = raw_arrs['startdate']
    enddate = raw_arrs['enddate']
    lens = raw_arrs['lens']
    train_arrs = {'sector_list': np.array([]),
            'lens': np.array([], dtype=np.int64),
            'conames': np.char.array([])}
    eval_arrs = train_arrs.copy()
    test_arrs = train_arrs.copy()
    arr_dicts = {'train': train_arrs, 'eval': eval_arrs, 'test': test_arrs}
    
    start = time.time()

    df_iterator = pd.read_csv(filepath + '.csv', index_col = [0,1], parse_dates=True, usecols = [0,2,3,4,5,6,7,8,9,10,11,12,13,15],
                 dtype = {'cheqv': np.float32, 'divd': np.float32,'divsp': np.float32,'cshtrd': np.float32,'prccd': np.float32,
                          'prcod': np.float32,'prchd': np.float32,'prcld': np.float32,'gsector': np.float32},
                 iterator = True)
    counters = [0,0,0]
    for nrows in lens:
        df = df_iterator.get_chunk(nrows)
        read += nrows
        dateindex = df.index.get_level_values(1)
        num_idx = datetime_to_number(dateindex, startdate)
        df.reset_index(level=1, inplace=True, drop=True)
        df.set_index(num_idx, append=True, inplace=True)
        train_start = 0
        train_end = enddate - constants.TEST_TIME - constants.VAL_TIME
        eval_start = train_end - constants.INPUT_DAYS
        eval_end = train_end + constants.VAL_TIME
        test_start = eval_end - constants.INPUT_DAYS
        test_end = enddate + 1
        datedict = {'train': (train_start, train_end),'eval': (eval_start, eval_end),'test': (test_start, test_end)}
        for counter_idx, label in enumerate(['train','eval','test']):
            processed_df, arrs = preprocess_chunk(df, (startdate, datedict[label]), arr_dicts[label], constants.MIN_DAYS_AVLB[counter_idx])
    
            if processed_df is None: continue
            counter_str = str(counters[counter_idx]).zfill(6)
            valid_name = f'{counter_str}_{"".join(x for x in arrs["conames"][-1] if x.isalnum())}'
            counters[counter_idx] += 1
            processed_df.to_csv(f'C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader/data/ccm3_processed/{label}/{valid_name}.csv')
        print(f'rows processed so far: {read}\ntime taken: {time.time() - start:.3f}', end = '\033[A\r', flush=True)
        
    print()

    for label in ['train','eval','test']:
        np.savez(f'C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader/data/ccm3_processed/{label}/identifiers.npz',
                 **arr_dicts[label])
        
    
if __name__ == '__main__':
    main()




