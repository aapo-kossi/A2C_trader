# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:04:41 2021

@author: Aapo Kössi
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import constants

def get_cats(lst):
    unique = pd.Series(lst).unique()[:-1]
    return unique

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

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))