# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:30:53 2020

@author: Aapo Kössi
"""

import glob
import csv
import time
import math
import numpy as np
import pandas as pd
import argparse

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


def get_cats(lst):
    unique = pd.Series(lst).unique()[:-1]
    return unique


def make_sliding_windows(ds, length):
    windows = ds.window(length, shift = constants.WINDOW_DIFF, drop_remainder = True)
    single_elem_windows = windows.map(lambda elem, names, secs: 
                                          (tf.data.experimental.get_single_element(
                                              elem.batch(length, drop_remainder = True)), 
                                              tf.data.experimental.get_single_element(names.take(1)),
                                              tf.data.experimental.get_single_element(secs.take(1))))                                             # WE OUT HERE TAKIN' NAMES
    return single_elem_windows


def window(batch, names, sectors, window_len = constants.WINDOW_LENGTH):
    
    tf.debugging.assert_all_finite(batch, 'unaligned batch of stocks not all finite values')
    s = tf.shape(batch)
    static_s = batch.shape
    day = batch[...,constants.others['data_index'].get_loc('date')]
    day = tf.cast(tf.reshape(day, [-1, ]), tf.int64)#.to_tensor()
    dayrow = tf.repeat(tf.range(s[0], dtype=tf.int64), s[1])
    idx = tf.stack([dayrow, day],axis=-1)
    # print(idx)
    # print(dayvals.shape)
    # print(dayrows.shape)
    # print(idx.shape)
    # batch = tf.sparse.SparseTensor(idx, batch.flat_values, [constants.others['n_stocks'], constants.TOTAL_TIME, constants.others['data_index'].size])
    batch = tf.scatter_nd(idx, tf.reshape(batch, [-1, s[2]]), [constants.others['n_stocks'], constants.others['enddate'] + 1, static_s[2]])
    batch = tf.transpose(batch, perm = [1,0,2])
    tf.debugging.assert_all_finite(batch, 'batch of stocks not all finite values')
    avlbl_idx = tf.where(tf.math.reduce_all(tf.math.reduce_any(batch != 0, axis = 2),axis=1))
    batch = tf.gather(batch, avlbl_idx)
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


def prepare_ds(ds, arrs, training = False, window_l = constants.WINDOW_LENGTH, n_envs = None, seed = None):
    ds = fetch_csvs(ds, arrs['lens'])
    ds = zip_identifiers(ds, arrs)
    if not training:
        ds = ds.repeat(16).shuffle(len(arrs['lens']))
        padded_l = arrs['lens'].max()
        ds = ds.map(lambda *x: pad_to_max_days(padded_l, *x), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(constants.others['n_stocks'], drop_remainder = True, num_parallel_calls = tf.data.AUTOTUNE)
        ds = ds.interleave(lambda *x: window(*x, window_len = window_l), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(128, seed = seed).take(n_envs).cache().repeat()
    else:
        ds = ds.cache().repeat().shuffle(len(arrs['lens']))
        padded_l = arrs['lens'].max()
        ds = ds.map(lambda *x: pad_to_max_days(padded_l, *x), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(constants.others['n_stocks'], drop_remainder = True, num_parallel_calls = tf.data.AUTOTUNE)
        ds = ds.interleave(window, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(constants.EP_SHUFFLE_BUF)
    return ds

def contains_special(string):
    bools = list(map(lambda char: char in constants.SPECIAL_CHARS, string))
    return any(bools)

def vis(state):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    while True:
        print(state.dates[:,0])
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


def apply_to_split(func):
    many = [func(label) for label in constants.SPLIT_LABELS]
    return tuple(many)

def load_ids(parent):
    load_id_func = lambda x: np.load(f'{parent}/{x}/identifiers.npz', allow_pickle=True)
    return apply_to_split(load_id_func)

def load_datasets(parent):
    def load_ds_func(x):
        globbed = f'{parent}/{x}/*.csv'
        names = glob.glob(globbed)
        return tf.data.Dataset.from_tensor_slices(names)
    
    return apply_to_split(load_ds_func)

def zip_identifiers(dataset, arrs):
    co_ds  = tf.data.Dataset.from_tensor_slices(arrs['conames'])
    sec_ds = tf.data.Dataset.from_tensor_slices(arrs['sector_list'].astype(np.float32))
    dataset = tf.data.Dataset.zip((dataset, co_ds, sec_ds))
    return dataset

def fetch_csvs(dataset, lens):
    def map_fn(l, filename):
        file_ds = tf.data.experimental.CsvDataset(filename, [tf.float32] * 7,
                                                  exclude_cols=[0, 7], header=True) #TODO: investigate buffer size
        file_ds = file_ds.batch(batch_size = l, drop_remainder=True).map(lambda *features: tf.stack(features, -1))
        return file_ds
    len_ds = tf.data.Dataset.from_tensor_slices(lens)
    dataset = tf.data.Dataset.zip((len_ds, dataset))
    dataset = dataset.interleave(map_fn, num_parallel_calls = tf.data.AUTOTUNE)
    return dataset
    
def pad_to_max_days(l, elem, name, sec):
    true_days = tf.shape(elem)[0]
    paddings = [[0,l - true_days],[0,0]]
    return tf.pad(elem, paddings), name, sec

def get_data_index(folderpath):
    filepath = f'{folderpath}/train/*.csv'
    first = next(glob.iglob(filepath))
    with open(first, 'r') as file:
        input_csv = csv.reader(file)
        cols = next(input_csv, [])
    data_index = pd.Index(cols)
    data_index = data_index.drop(['gsector', 'GVKEY'])
    return data_index

def get_enddate(folderpath):
    npz = np.load(f'{folderpath}/ccm3_raw_lens.npz', allow_pickle=True)
    return npz['enddate']
        

def main():
    
    print('started')
    plt.ion()
    
    


    parser = argparse.ArgumentParser(description='Train a neural network to trade n stocks concurrently, '\
                                                 'provided a .csv file of stock data.')

    parser.add_argument('-d', '--input_dir', help = 'path to the dir including processed csv split into train, eval and test dirs', type=str)
    parser.add_argument('-n', '--num_stocks', help = 'number of stocks the model is to have as an input', type=str)
    args = parser.parse_args()

    if args.num_stocks is not None:
        constants.add('n_stocks', args.num_stocks)
    else: constants.add('n_stocks', constants.DEFAULT_TICKERS)

    parentdir = args.input_dir
    constants.add('data_index', get_data_index(parentdir))
    constants.add('enddate', get_enddate(parentdir))
    
    train_arrs, eval_arrs, test_arrs = load_ids(parentdir)
    sec_cats = get_cats(train_arrs['sector_list'])
    train_ds, eval_ds, test_ds = load_datasets(parentdir)
    train_ds = prepare_ds(train_ds, train_arrs, training=True)
    
    eval_ds = prepare_ds(eval_ds, eval_arrs,
                         window_l = constants.INPUT_DAYS + constants.VAL_STEPS,
                         n_envs = constants.N_VAL_ENVS, seed = 0)
    
    test_ds = prepare_ds(test_ds, test_arrs,
                         window_l = constants.INPUT_DAYS + constants.TEST_STEPS,
                         n_envs = constants.N_TEST_ENVS, seed = 1)


    # start = time.time()
    # for n, elem in enumerate(test_ds):
    #     print(n)
    #     taken = time.time() - start
    #     print(f'current fps: {n / taken}')
    #     # tf.print(elem, summarize = -1)
    #     # break
    # raise SystemExit

    #visualized train_windows of stock performances

    # mock_env = TradingEnv(train_ds,
    #               constants.others['data_index'],
    #               sec_cats,
    #               tf.constant((constants.others['n_stocks'],), dtype = tf.int32),
    #               noise_ratio = 0.0)
    
    # vis(mock_env)


    
    # initialize envs and model
    output_shape = tf.constant((constants.others['n_stocks']), dtype = tf.int32)   
    vec_trading_env = TradingEnv(train_ds, constants.others['data_index'],
                                sec_cats, (output_shape,),
                                n_envs = constants.N_ENVS,
                                init_capital = 50000, MAR = constants.RF, noise_ratio=constants.NOISE_RATIO)
   
    eval_env = TradingEnv(eval_ds, constants.others['data_index'], sec_cats,
                          (output_shape,),n_envs= constants.N_VAL_ENVS, init_capital=50000, MAR = constants.RF, noise_ratio= 0.0)

    test_env = TradingEnv(test_ds, constants.others['data_index'], sec_cats,
                          (output_shape,),n_envs= constants.N_TEST_ENVS, init_capital=50000, MAR = constants.RF, noise_ratio= 0.0)

    model = TradingModel.Trader(output_shape)

    # learn using a2c algorithm
    learn(model,
          vec_trading_env,
          val_env = eval_env,
          test_env = test_env,
          steps_per_update=constants.N_STEPS_UPDATE,
          eval_steps=constants.VAL_TIME,
          test_steps=constants.TEST_TIME,
          init_lr = constants.INIT_LR,
          decay_steps = constants.INIT_DECAY_STEPS,
          decay_rate= constants.DECAY_RATE,
          t_mul = constants.T_MUL,
          m_mul = constants.M_MUL,
          lr_alpha = constants.LR_ALPHA,
          gamma=constants.GAMMA)
    
    
if __name__ == '__main__':
    main()