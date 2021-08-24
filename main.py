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
import tensorflow as tf
import keras_tuner as kt
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(918274)

from matplotlib import pyplot as plt
from gym_tf_env import TradingEnv
from a2c import learn
from tuner import MyTuner
from tfrecords import write_record
import TradingModel
import constants





def get_cats(lst):
    unique = pd.Series(lst).unique()[:-1]
    return unique


def make_sliding_windows(ds, length):
    windows = ds.window(length, shift = constants.WINDOW_DIFF, drop_remainder = True)
    single_elem_windows = windows.interleave(lambda elem, names, secs: 
                                             tf.data.Dataset.zip(
                                             (elem.batch(length, drop_remainder = True), 
                                              names.take(1),
                                              secs.take(1))), num_parallel_calls = tf.data.AUTOTUNE)                                             # WE OUT HERE TAKIN' NAMES
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

#TODO: consider revert to iterating indefinitely
def prepare_ds(ds, arrs, training = False, seed = None):    

    ds = fetch_csvs(ds, arrs['lens'])
    ds = zip_identifiers(ds, arrs)
    if not training:
        ds = ds.repeat(16).shuffle(len(arrs['lens']), seed = seed)
        padded_l = arrs['lens'].max()
        ds = ds.map(lambda *x: pad_to_max_days(padded_l, *x), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(constants.others['n_stocks'], drop_remainder = True, num_parallel_calls = tf.data.AUTOTUNE)
    else:
        ds = ds.cache().repeat().shuffle(len(arrs['lens']))
        padded_l = arrs['lens'].max()
        ds = ds.map(lambda *x: pad_to_max_days(padded_l, *x), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(constants.others['n_stocks'], drop_remainder = True, num_parallel_calls = tf.data.AUTOTUNE)
    return ds

def finish_ds(ds, arrs, training = False, window_l = constants.WINDOW_LENGTH, n_envs = None, seed = None):
    if not training:
        ds = ds.interleave(lambda *x: window(*x, window_len = window_l), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(128, seed = seed).take(n_envs).cache().repeat()
    else:
        ds = ds.interleave(window, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(constants.EP_SHUFFLE_BUF)
        # ds = ds.take(8192).cache().repeat().shuffle(8192)
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
            plt.pause(2)
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
                                                  exclude_cols=[0, 7], header=True, buffer_size = 1048576) # buffer size 1 MB
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
    len_path = glob.glob(f'{folderpath}/*_metadata.npz')[0]
    npz = np.load(len_path, allow_pickle=True)
    return npz['enddate']

def setup_metrics():
    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype = tf.float64)
    train_pg_loss_metric = tf.keras.metrics.Mean('train_pg_loss', dtype = tf.float64)
    train_val_loss_metric = tf.keras.metrics.Mean('train_value_loss', dtype = tf.float64)
    train_ent_metric = tf.keras.metrics.Mean('train_ent', dtype = tf.float64)
    train_reward_metric = tf.keras.metrics.Mean('train_reward', dtype = tf.float64)

    eval_loss_metric = tf.keras.metrics.Mean('eval_loss', dtype = tf.float64)
    eval_pg_loss_metric = tf.keras.metrics.Mean('eval_pg_loss', dtype = tf.float64)
    eval_val_loss_metric = tf.keras.metrics.Mean('eval_value_loss', dtype = tf.float64)
    eval_ent_metric = tf.keras.metrics.Mean('eval_ent', dtype = tf.float64)
    eval_reward_metric = tf.keras.metrics.Mean('eval_reward', dtype = tf.float64)


    test_loss_metric = tf.keras.metrics.Mean('test_loss', dtype = tf.float64)
    test_pg_loss_metric = tf.keras.metrics.Mean('test_pg_loss', dtype = tf.float64)
    test_val_loss_metric = tf.keras.metrics.Mean('test_value_loss', dtype = tf.float64)
    test_ent_metric = tf.keras.metrics.Mean('test_ent', dtype = tf.float64)
    test_reward_metric = tf.keras.metrics.Mean('test_reward', dtype = tf.float64)
    metrics = {'train_rew': train_reward_metric,
               'train_ent': train_ent_metric,
               'train_value_loss': train_val_loss_metric,
               'train_pg_loss': train_pg_loss_metric,
               'train_loss': train_loss_metric,
               'eval_rew': eval_reward_metric,
               'eval_ent': eval_ent_metric,
               'eval_value_loss': eval_val_loss_metric,
               'eval_pg_loss': eval_pg_loss_metric,
               'eval_loss': eval_loss_metric,
               'test_rew': test_reward_metric,
               'test_ent': test_ent_metric,
               'test_value_loss': test_val_loss_metric,
               'test_pg_loss': test_pg_loss_metric,
               'test_loss': test_loss_metric,}
    return metrics


def main():
    
    print('started')
    plt.ion()
    
    


    parser = argparse.ArgumentParser(description='Train a neural network to trade n stocks concurrently, '\
                                                 'provided a .csv file of stock data.')

    # parser.add_argument('save_path', help = 'path to the folder used to save model weights as checkpoints', type = str)
    parser.add_argument('-d', '--input_dir', help = 'path to the dir including processed csv split into train, eval and test dirs', type=str)
    parser.add_argument('-n', '--num_stocks', help = 'number of stocks the model is to have as an input', type=int)
    parser.add_argument('-c', '--checkpoint', help = 'specify a checkpoint file to load weights from, if unspecified, will initiate model from scratch', type = str)
    parser.add_argument('-l', '--use_latest', help = 'use latest checkpoint to continue training', action = 'store_true')

    args = parser.parse_args()

    if args.num_stocks is not None:
        constants.add('n_stocks', args.num_stocks)
    else: constants.add('n_stocks', constants.DEFAULT_TICKERS)

    parentdir = args.input_dir
    constants.add('data_index', get_data_index(parentdir))
    constants.add('enddate', get_enddate(parentdir))


    # init_ckpt = args.checkpoint
    # if args.use_latest: init_ckpt = 'latest'
    
    train_arrs, eval_arrs, test_arrs = load_ids(parentdir)
    sec_cats = get_cats(train_arrs['sector_list'])
    train_ds, eval_ds, test_ds = load_datasets(parentdir)
    
    
    metrics = setup_metrics()
    
    hypermodel = TradingModel.HyperTrader(out_shape = constants.others['n_stocks'])
    
    hp = kt.HyperParameters()
    hp.Fixed('temporal_nn_type', value = 'LSTM')
    hp.Fixed('max_steps_env', value = 32)
    hp.Fixed('n_steps_update', value = 12)
    hp.Fixed('n_batch', value = 16)
    oracle = kt.oracles.Hyperband(kt.Objective('fitness','max'), max_epochs = 50, hyperparameters = hp)
    
        
    tuner = MyTuner(oracle, hypermodel, summary_metrics = metrics, project_name = 'trader_optimization', n_stocks = constants.others['n_stocks'],
                    sec_cats = sec_cats, train_arrs = train_arrs,
                    eval_arrs = eval_arrs, test_arrs = test_arrs,
                    # save_path = args.save_path,
                    data_index = constants.others['data_index'])
    

    train_ds = prepare_ds(train_ds, train_arrs, training=True)
    eval_ds = prepare_ds(eval_ds, eval_arrs)
    test_ds = prepare_ds(test_ds, test_arrs)    
    
    # tuner.search((train_ds, eval_ds, test_ds))
    
    # best_hparams = tuner.get_best_hyperparameters()
    # print(best_hparams)
    
    # train_ds = prepare_ds(train_ds, train_arrs, training=True)
    train_ds = finish_ds(train_ds, train_arrs, training = True,
                          n_envs = constants.N_ENVS)
    write_record(train_ds, 'C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader/data/ccm4_processed/train')
    
    
    # eval_ds = prepare_ds(eval_ds, eval_arrs, seed = 0)
    eval_ds = finish_ds(eval_ds, eval_arrs,
                        window_l = constants.INPUT_DAYS + constants.VAL_STEPS,
                        n_envs = constants.N_VAL_ENVS, seed = 0)
    write_record(eval_ds, 'C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader/data/ccm4_processed/eval')
    
    # test_ds = prepare_ds(test_ds, test_arrs, seed = 1)
    test_ds = finish_ds(test_ds, test_arrs,
                        window_l = constants.INPUT_DAYS + constants.TEST_STEPS,
                        n_envs = constants.N_TEST_ENVS, seed = 1)
    write_record(test_ds, 'C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader/data/ccm4_processed/test')


    # start = time.time()
    # while time.time() - start < 900:
    #     for n, elem in enumerate(train_ds):
    #         print(n)
    #         taken = time.time() - start
    #         print(f'current fps: {n / taken}')
    #         # tf.print(elem, summarize = -1)
    #     # break
    # print(f'elements fetched in 15 minutes: {n}')
    # raise SystemExit

    # visualized train_windows of stock performances

    # mock_env = TradingEnv(train_ds,
    #               constants.others['data_index'],
    #               sec_cats,
    #               tf.constant((constants.others['n_stocks'],), dtype = tf.int32),
    #               noise_ratio = 0.0)
    
    # vis(mock_env)


    
    # initialize envs and model
    # output_shape = tf.constant((constants.others['n_stocks']), dtype = tf.int32)   
    # vec_trading_env = TradingEnv(train_ds, constants.others['data_index'],
    #                             sec_cats, (output_shape,),
    #                             n_envs = constants.N_ENVS,
    #                             init_capital = 50000, MAR = constants.RF, noise_ratio=constants.NOISE_RATIO,
    #                             cost_per_share=constants.COST_PER_SHARE)
   
    # eval_env = TradingEnv(eval_ds, constants.others['data_index'], sec_cats,
    #                       (output_shape,),n_envs= constants.N_VAL_ENVS, init_capital=50000, MAR = constants.RF, noise_ratio= 0.0,
    #                       cost_per_share=constants.COST_PER_SHARE)

    # test_env = TradingEnv(test_ds, constants.others['data_index'], sec_cats,
    #                       (output_shape,),n_envs= constants.N_TEST_ENVS, init_capital=50000, MAR = constants.RF, noise_ratio= 0.0,
    #                       cost_per_share=constants.COST_PER_SHARE)

    # model = TradingModel.Trader(output_shape)

    # # learn using a2c algorithm
    # learn(model,
    #       vec_trading_env,
    #       args.save_path,
    #       initial_ckpt = init_ckpt,
    #       val_env = eval_env,
    #       test_env = test_env,
    #       steps_per_update=constants.N_STEPS_UPDATE,
    #       eval_steps=constants.VAL_TIME,
    #       test_steps=constants.TEST_TIME,
    #       init_lr = constants.INIT_LR,
    #       decay_steps = constants.INIT_DECAY_STEPS,
    #       decay_rate= constants.DECAY_RATE,
    #       t_mul = constants.T_MUL,
    #       m_mul = constants.M_MUL,
    #       lr_alpha = constants.LR_ALPHA,
    #       gamma=constants.GAMMA)
    
    
if __name__ == '__main__':
    main()