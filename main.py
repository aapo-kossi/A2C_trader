# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:30:53 2020

@author: Aapo Kössi
"""

import time
import math
import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass
from typing import List

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from matplotlib import pyplot as plt
from gym_tf_env import TradingEnv
from a2c import learn
import TradingModel
import constants
#accesses datasets, starts training pipeline, monitors progress

@dataclass
class FileData:
    dataset: tf.RaggedTensor
    data_index: pd.Index
    tickers: List[str]
    sectors: List[float]
    sec_cats: np.array
    num_tickers: int
    lens: List[int]
    


def get_cats(lst):
    unique = pd.Series(lst).unique()[:-1]
    return unique

def datetime_to_number(dt_index):
    first = dt_index.min()
    numerical_index = dt_index.map(lambda val: (val - first).days.astype('float'))
    tot_days = numerical_index.max().astype('int')
    constants.others['TOTAL_DAYS'] = tot_days
    return numerical_index

def split_ds(filedata):

    dateidx = filedata.data_index.get_loc('date')
    enddate = constants.others['TOTAL_DAYS']
    overlap_days = constants.INPUT_DAYS
    test_startdate = enddate - (overlap_days + constants.TEST_TIME)
    val_enddate = enddate - constants.TEST_TIME
    val_startdate = test_startdate - constants.VAL_TIME
    train_enddate = val_startdate + overlap_days

    train_mask = filedata.dataset[...,dateidx] < train_enddate
    train_ds = tf.ragged.boolean_mask(filedata.dataset,train_mask)
    co_ds = tf.data.Dataset.from_tensor_slices(filedata.tickers)
    sec_ds = tf.data.Dataset.from_tensor_slices(tf.constant(filedata.sectors, dtype=tf.float32))
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    train_ds = tf.data.Dataset.zip((train_ds, co_ds, sec_ds))
    train_ds = train_ds.repeat().shuffle(8192).batch(constants.others['n_stocks'],\
                                                                            drop_remainder=True, num_parallel_calls = tf.data.AUTOTUNE)

    val_mask = tf.logical_and(filedata.dataset[...,dateidx] >= val_startdate, filedata.dataset[...,dateidx] < val_enddate)
    val_ds = tf.ragged.boolean_mask(filedata.dataset,val_mask)
    val_ds = tf.data.Dataset.from_tensor_slices(val_ds)
    val_ds = tf.data.Dataset.zip((val_ds, co_ds, sec_ds))
    val_ds = val_ds.repeat(16).shuffle(2048, seed=0).batch(constants.others['n_stocks'], drop_remainder=True)

    test_mask= filedata.dataset[...,dateidx] >= test_startdate
    test_ds = tf.ragged.boolean_mask(filedata.dataset,test_mask)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = tf.data.Dataset.zip((test_ds, co_ds, sec_ds))
    test_ds = test_ds.repeat(16).shuffle(2048, seed=1).batch(constants.others['n_stocks'], drop_remainder=True)
    
    # daily_ds = ds.batch(n_tickers, drop_remainder=True)
    
    # train_ds = daily_ds.take(int(constants.TRAIN_TIME * constants.TOTAL_TIME))
    # val_ds   = daily_ds.skip(int(constants.TRAIN_TIME * constants.TOTAL_TIME)-constants.INPUT_DAYS)
    # val_ds   = val_ds.take(constants.INPUT_DAYS + int(constants.VAL_TIME * constants.TOTAL_TIME))
    # test_ds  = daily_ds.skip(int(constants.TOTAL_TIME * (1 - constants.TEST_TIME)) - constants.INPUT_DAYS)
    # test_ds  = test_ds.take(constants.INPUT_DAYS + int(constants.TEST_TIME * constants.TOTAL_TIME))
    return train_ds, val_ds, test_ds



def align_ragged_dates(values, index):
    pass

def make_sliding_windows(ds, length):
    windows = ds.window(length, shift = constants.WINDOW_DIFF, drop_remainder = True)
    single_elem_windows = windows.map(lambda elem, names, secs: 
                                          (tf.data.experimental.get_single_element(
                                              elem.batch(length, drop_remainder = True)), 
                                              tf.data.experimental.get_single_element(names.take(1)),
                                              tf.data.experimental.get_single_element(secs.take(1))))                                             # WE OUT HERE TAKIN' NAMES
    return single_elem_windows

def filter_ds(ds):
    ds = ds.filter(lambda x, names, secs: tf.math.reduce_all(tf.math.reduce_any(x != 0, axis=2)))
    return ds
    



def window(batch, names, sectors, window_len = constants.WINDOW_LENGTH):
    
    tf.debugging.assert_all_finite(batch.to_tensor(), 'unaligned batch of stocks not all finite values')
    days = batch[...,constants.others['data_index'].get_loc('date')]
    dayvals = tf.cast(days.values, tf.int64)#.to_tensor()
    dayrows = days.value_rowids()#.to_tensor()
    idx = tf.stack([dayrows, dayvals],axis=-1)
    # print(idx)
    # print(dayvals.shape)
    # print(dayrows.shape)
    # print(idx.shape)
    # batch = tf.sparse.SparseTensor(idx, batch.flat_values, [constants.others['n_stocks'], constants.TOTAL_TIME, constants.others['data_index'].size])
    batch = tf.scatter_nd(idx, batch.flat_values, [constants.others['n_stocks'], constants.others['TOTAL_DAYS'] + 1, constants.others['data_index'].size])
    batch = tf.transpose(batch, perm = [1,0,2])
    tf.debugging.assert_all_finite(batch, 'batch of stocks not all finite values')
    batch = tf.gather(batch, tf.where(tf.math.reduce_all(tf.math.reduce_any(batch != 0, axis = 2),axis=1)))
    batch = tf.squeeze(batch, axis = 1)
    batch_ds = tf.data.Dataset.from_tensor_slices(batch)
    names_repeated = tf.repeat(tf.expand_dims(names, 0), tf.shape(batch)[0], axis = 0)
    secs_repeated = tf.repeat(tf.expand_dims(sectors, 0), tf.shape(batch)[0], axis = 0)
    name_ds = tf.data.Dataset.from_tensor_slices(names_repeated)
    sec_ds = tf.data.Dataset.from_tensor_slices(secs_repeated)
    labeled_ds = tf.data.Dataset.zip((batch_ds, name_ds, sec_ds))
    #split elem of these stocks into timestep windows
    labeled_ds = labeled_ds.apply(lambda x: make_sliding_windows(x, window_len))
    #keep windows that have stock data for all stocks on the same days, drop otherwise
    return labeled_ds

def contains_special(string):
    bools = list(map(lambda char: char in constants.SPECIAL_CHARS, string))
    return any(bools)

def preprocess(df):
    df = df[df['curcdd']=='USD']
    df.drop(['curcdd'], axis='columns', inplace=True)
    print(df.head())
    df['ajexdi'] = df['ajexdi'].replace(0,1)
    df[['prccd','prchd','prcld','prcod','divd','divsp']] = \
        df[['prccd','prchd','prcld','prcod','divd','divsp']].multiply(
        df['ajexdi'] ** -1,axis = 'index')
    df.fillna(0, inplace=True)
    df['dist'] = df['cheqv'] + df['divd'] + df['divsp']
    df.drop(['ajexdi','cheqv','divd','divsp'],axis='columns', inplace=True)
    print('did initial preprocessing')
    ticker_dfs = list(df.groupby('GVKEY', sort=False))
    del df
    nested_datalists= []
    sector_list = []
    lens = []
    conames = []
    print('starting loop')
    for gvkey, df in ticker_dfs:
        if df.shape[0] < constants.MIN_DAYS_AVLB: continue
        # min_vol = df['cshtrd'].min()
        # if min_vol < constants.MIN_MIN_VOL: continue
        avg_vol = df['cshtrd'].sum() / df.shape[0]
        if avg_vol < constants.MIN_AVG_VOL: continue
        # print('enough datapoints to include')
        df.sort_index(inplace=True)
        sector_list.append(df.gsector.iloc[0])
        conames.append(df.conm.iloc[0])
        df.drop(['conm'],axis=1,inplace = True)
        unique = ~df.index.duplicated(keep='first')
        df = df[unique]
        df.reset_index(level=1, inplace=True)
        df.rename({'datadate':'date'},axis=1,inplace=True)
        # df['date'] = df['date'].astype('float')
        # print('dropped duplicate days')
        nested_datalists.append(df)
        lens.append(df.shape[0])
    print('finished loop')
    # df = pd.concat(processed_dfs).reset_index(level = 1)
    print(lens)
    print(len(lens))
    print('got complete nested data')
    df = pd.concat(nested_datalists)
    df.drop(['gsector'], axis='columns', inplace=True)
    data_index = df.columns
    dataset = tf.constant(df.to_numpy(), dtype = tf.float32)
    del df
    tf.debugging.assert_all_finite(dataset, 'original original tensor not finite')
    dataset = tf.RaggedTensor.from_row_lengths(values = dataset, row_lengths = lens)    
    sec_cats = get_cats(sector_list)
    num_tickers = len(conames)
    complete_data = FileData(dataset, data_index, conames, sector_list, sec_cats, num_tickers, lens)
    return complete_data
    

def vis(state):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    while True:
        print(state.ohlcvd[0,:4,0,0])
        for i in range(constants.N_ENVS):
            ydata = state.ohlcvd[i,...,constants.others['data_index'].get_loc('prccd')]
            for j in range(constants.others['n_stocks']):
                wavelength = constants.others['n_stocks']
                omega = 2*math.pi*j/wavelength
                r_val = 0.5*(1 + math.sin(omega))
                g_val = 0.5*(1 + math.sin(omega + 2/3*math.pi))
                b_val = 0.5*(1 + math.sin(omega + 4/3*math.pi))
                color = [r_val, g_val, b_val]
                y = ydata[:,j]
                x = range(y.shape[0])
                ax.plot(x, y,
                        label = state.conames[i,j].numpy(),
                        color = color)
            ax.legend()
            plt.pause(10)
            ax.cla()
            state.reset()


def main():
    
    print('started')
    plt.ion()
    
    


    parser = argparse.ArgumentParser(description='Train a neural network to trade n stocks concurrently, '\
                                                 'provided a .csv file of stock data.')

    parser.add_argument('-f', '--filepath', help = 'path of the .csv input file', type=str)
    parser.add_argument('-n', '--num_stocks', help = 'number of stocks the model is to have as an input', type=str)
    args = parser.parse_args()

    filepath = args.filepath

    df = pd.read_csv(filepath, index_col = [0,1], parse_dates=True, usecols = [0,2,3,4,5,6,7,8,9,10,11,12,13,15],
                     dtype = {'cheqv': np.float32, 'divd': np.float32,'divsp': np.float32,'cshtrd': np.float32,'prccd': np.float32,
                              'prcod': np.float32,'prchd': np.float32,'prcld': np.float32,'gsector': np.float32})
    dateindex = df.index.get_level_values(1)
    num_idx = datetime_to_number(dateindex)
    df.reset_index(level=1, inplace=True, drop=True)
    df.set_index(num_idx, append=True, inplace=True)
    df.astype(np.float32, copy=False, errors='ignore')
    complete_data = preprocess(df)
        
    #TODO: can't imagine next line follows any sort of best practices
    constants.add('data_index', complete_data.data_index)
    if args.num_stocks is not None:
        constants.add('n_stocks', args.num_stocks)
    else: constants.add('n_stocks', constants.DEFAULT_TICKERS)

    tf.debugging.assert_all_finite(complete_data.dataset.to_tensor(default_value=0), 'original tensor not finite')
    train_ds, val_ds, test_ds = split_ds(complete_data)
    train_ds = train_ds.flat_map(window).prefetch(32)
    # map(window, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda *x: tf.data.Dataset.zip(*x)).prefetch(32)

    
    
    val_ds = val_ds.flat_map(lambda data, names, sectors: window(data, names, sectors, window_len=constants.INPUT_DAYS + constants.VAL_STEPS))
    val_ds = val_ds.shuffle(32, seed = 2).take(constants.N_VAL_ENVS).cache().repeat()
    test_ds = test_ds.flat_map(lambda data, names, sectors: window(data, names, sectors, window_len=constants.INPUT_DAYS + constants.TEST_STEPS))
    test_ds = test_ds.shuffle(32, seed = 1).take(constants.N_TEST_ENVS).cache().repeat()
    
    #organize the training dataset into shuffled windows
    # train_ds = train_ds.shuffle(constants.EP_SHUFFLE_BUF)
    eval_steps = constants.VAL_TIME
    test_steps = constants.TEST_TIME
    # val_ds = val_ds.apply(lambda x: make_sliding_windows(x, constants.INPUT_DAYS + eval_steps))
    # test_ds = test_ds.apply(lambda x: make_sliding_windows(x, constants.INPUT_DAYS + test_steps))

    # reversed_ticker_dict = {float(value) : key for (key, value) in complete_data.ticker_dict.items()}


    
    # start = time.time()
    # for _ in val_ds:
    #     print(time.time() - start)

    #visualized train_windows of stock performances

    # mock_env = TradingEnv(test_ds,
    #               complete_data.data_index,
    #               complete_data.sec_cats,
    #               tf.constant((constants.others['n_stocks'],), dtype = tf.int32),
    #               noise_ratio = 0.0)
    
    # vis(mock_env)


    
    # initialize envs and model
    output_shape = tf.constant((constants.others['n_stocks']), dtype = tf.int32)   
    vec_trading_env = TradingEnv(train_ds, complete_data.data_index,
                                complete_data.sec_cats, (output_shape,),
                                n_envs = constants.N_ENVS,
                                init_capital = 50000, MAR = constants.RF, noise_ratio=constants.NOISE_RATIO)
   
    val_env = TradingEnv(val_ds, complete_data.data_index, complete_data.sec_cats,
                          (output_shape,),n_envs= constants.N_VAL_ENVS, init_capital=50000, MAR = constants.RF, noise_ratio=constants.NOISE_RATIO)
    model = TradingModel.Trader(output_shape)

    # learn using a2c algorithm
    learn(model,
          vec_trading_env,
          val_env = val_env,
          steps_per_update=constants.N_STEPS_UPDATE,
          eval_steps=int(eval_steps),
          test_steps=int(test_steps),
          init_lr = constants.INIT_LR,
          decay_steps = constants.INIT_DECAY_STEPS,
          decay_rate= constants.DECAY_RATE,
          t_mul = constants.T_MUL,
          m_mul = constants.M_MUL,
          lr_alpha = constants.LR_ALPHA,
          gamma=constants.GAMMA)
    
    
if __name__ == '__main__':
    main()