# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:04:41 2021

@author: Aapo Kössi
"""

import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import tensorflow as tf
import glob
import csv
import zipfile
import constants


def load_processed_datasets(path, arrs, date_col):
    train, val, test = load_datasets(path)
    train = tf.data.experimental.load(f'{path}/train').repeat().shuffle(1024)

    val = prepare_ds(val, arrs[1], seed=0)
    val = finish_ds(val, arrs[1], date_col,
                    window_l=constants.INPUT_DAYS + constants.VAL_STEPS,
                    n_envs=constants.N_VAL_ENVS, seed=0)
    val = val.map(hasher)

    test = prepare_ds(test, arrs[2], seed=1)
    test = finish_ds(test, arrs[2], date_col,
                     window_l=constants.INPUT_DAYS + constants.TEST_STEPS,
                     n_envs=constants.N_TEST_ENVS, seed=1)
    test = test.map(hasher)

    return train, val, test


def gen_train(path, arr, date_col):
    ds, _, _ = load_datasets(path)
    ds = prepare_ds(ds, arr, training=True)
    ds = finish_ds(ds, arr, date_col, training=True)
    return ds


def save_train(ds, path):
    shard = tf.Variable(initial_value=-1, dtype=tf.int64)

    def shard_func(*_):
        if shard == 1:
            return shard.assign(0)
        else:
            return tf.convert_to_tensor(shard.assign_add(1), dtype=tf.int64)

    tf.data.experimental.save(ds.take(2 ** 18), f'{path}/train',
                              shard_func=shard_func)


def contains_special(string):
    bools = list(map(lambda char: char in constants.SPECIAL_CHARS, string))
    return any(bools)


def vis(state):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    while True:
        print(state.dates[:, 0])
        for i in range(constants.N_ENVS):
            ydata = state.ohlcvd[i, ..., state.window_data_index.get_loc('prccd')]
            for j in range(constants.DEFAULT_TICKERS):
                wavelength = constants.DEFAULT_TICKERS
                omega = 2 * math.pi * j / wavelength
                r_val = 0.5 * (1 + math.sin(omega))
                g_val = 0.5 * (1 + math.sin(omega + 2 / 3 * math.pi))
                b_val = 0.5 * (1 + math.sin(omega + 4 / 3 * math.pi))
                color = [r_val, g_val, b_val]
                y = ydata[:, j]
                x = range(y.shape[0])
                ax.plot(x, y,
                        label=state.conames[i, j].numpy(),
                        color=color)
            ax.legend()
            plt.pause(1)
            ax.cla()
            state.reset()


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
    co_ds = tf.data.Dataset.from_tensor_slices(arrs['conames'])
    sec_ds = tf.data.Dataset.from_tensor_slices(arrs['sector_list'].astype(np.float32))
    dataset = tf.data.Dataset.zip((dataset, co_ds, sec_ds))
    return dataset


def fetch_csvs(dataset, lens):
    def map_fn(l, filename):
        file_ds = tf.data.experimental.CsvDataset(filename, [tf.float32] * 7,
                                                  exclude_cols=[0, 7], header=True,
                                                  buffer_size=1048576)  # buffer size 1 MB
        file_ds = file_ds.batch(batch_size=l, drop_remainder=True).map(lambda *features: tf.stack(features, -1))
        return file_ds

    len_ds = tf.data.Dataset.from_tensor_slices(lens)
    dataset = tf.data.Dataset.zip((len_ds, dataset))
    dataset = dataset.interleave(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def pad_to_max_days(l, elem, name, sec):
    true_days = tf.shape(elem)[0]
    paddings = [[0, l - true_days], [0, 0]]
    return tf.pad(elem, paddings), name, sec


def make_sliding_windows(ds, length):
    windows = ds.window(length, shift=constants.WINDOW_DIFF, drop_remainder=True)
    single_elem_windows = windows.interleave(lambda elem, names, secs:
                                             tf.data.Dataset.zip(
                                                 (elem.batch(length, drop_remainder=True),
                                                  names.take(1),  # WE OUT HERE TAKIN' NAMES
                                                  secs.take(1))), num_parallel_calls=tf.data.AUTOTUNE)
    return single_elem_windows


def window(batch, names, sectors, date_col, window_len=constants.WINDOW_LENGTH):
    s = tf.shape(batch)
    static_s = batch.shape
    day = batch[..., date_col]
    day = tf.cast(tf.reshape(day, [-1, ]), tf.int64)  # .to_tensor()
    dayrow = tf.repeat(tf.range(s[0], dtype=tf.int64), s[1])
    idx = tf.stack([dayrow, day], axis=-1)
    batch = tf.scatter_nd(idx, tf.reshape(batch, [-1, s[2]]), [constants.DEFAULT_TICKERS, 2 ** 14, static_s[2]])
    batch = tf.transpose(batch, perm=[1, 0, 2])
    avlbl_idx = tf.where(tf.math.reduce_all(tf.math.reduce_any(batch != 0, axis=2), axis=1))
    batch = tf.gather(batch, avlbl_idx)
    batch = tf.squeeze(batch, axis=1)
    batch_ds = tf.data.Dataset.from_tensor_slices(batch)
    names_repeated = tf.repeat(tf.expand_dims(names, 0), tf.shape(batch)[0], axis=0)
    secs_repeated = tf.repeat(tf.expand_dims(sectors, 0), tf.shape(batch)[0], axis=0)
    name_ds = tf.data.Dataset.from_tensor_slices(names_repeated)
    sec_ds = tf.data.Dataset.from_tensor_slices(secs_repeated)
    labeled_ds = tf.data.Dataset.zip((batch_ds, name_ds, sec_ds))
    # split elem of these stocks into timestep windows
    labeled_ds = labeled_ds.apply(lambda x: make_sliding_windows(x, window_len))
    # keep windows that have stock data for all stocks on the same days, drop otherwise
    return labeled_ds


# TODO: consider revert to iterating indefinitely
def prepare_ds(ds, arrs, training=False, seed=None):
    ds = fetch_csvs(ds, arrs['lens'])
    ds = zip_identifiers(ds, arrs)
    if not training:
        ds = ds.repeat(16).shuffle(len(arrs['lens']), seed=seed)
        padded_l = arrs['lens'].max()
        ds = ds.map(lambda *x: pad_to_max_days(padded_l, *x), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(constants.DEFAULT_TICKERS, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.cache().repeat().shuffle(len(arrs['lens']))
        padded_l = arrs['lens'].max()
        ds = ds.map(lambda *x: pad_to_max_days(padded_l, *x), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(constants.DEFAULT_TICKERS, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def finish_ds(ds, arrs, date_col, training=False, window_l=constants.WINDOW_LENGTH, n_envs=None, seed=None):
    if not training:
        ds = ds.interleave(lambda *x: window(*x, date_col, window_len=window_l), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(128, seed=seed).take(n_envs).cache().repeat()
    else:
        def predicate(*elem):
            ohlcvd = elem[0]
            days = ohlcvd[:, 0, date_col]
            diffs = days[1:] - days[:-1]
            dense = tf.reduce_all(diffs <= 30)
            return dense

        ds = ds.interleave(lambda *x: window(*x, date_col), num_parallel_calls=tf.data.AUTOTUNE)
        ds.filter(predicate).prefetch(tf.data.AUTOTUNE)
        # ds = ds.take(8192).cache().repeat().shuffle(8192)
    return ds


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_data_index(folderpath):
    filepath = f'{folderpath}/test/*.csv'
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
    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_pg_loss_metric = tf.keras.metrics.Mean('train_pg_loss', dtype=tf.float32)
    train_val_loss_metric = tf.keras.metrics.Mean('train_value_loss', dtype=tf.float32)
    train_ent_metric = tf.keras.metrics.Mean('train_ent', dtype=tf.float32)
    train_reward_metric = tf.keras.metrics.Mean('train_reward', dtype=tf.float32)

    eval_loss_metric = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
    eval_pg_loss_metric = tf.keras.metrics.Mean('eval_pg_loss', dtype=tf.float32)
    eval_val_loss_metric = tf.keras.metrics.Mean('eval_value_loss', dtype=tf.float32)
    eval_ent_metric = tf.keras.metrics.Mean('eval_ent', dtype=tf.float32)
    eval_reward_metric = tf.keras.metrics.Mean('eval_reward', dtype=tf.float32)

    test_loss_metric = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_pg_loss_metric = tf.keras.metrics.Mean('test_pg_loss', dtype=tf.float32)
    test_val_loss_metric = tf.keras.metrics.Mean('test_value_loss', dtype=tf.float32)
    test_ent_metric = tf.keras.metrics.Mean('test_ent', dtype=tf.float32)
    test_reward_metric = tf.keras.metrics.Mean('test_reward', dtype=tf.float32)
    train_metrics = {'train_rew': train_reward_metric,
                     'train_ent': train_ent_metric,
                     'train_value_loss': train_val_loss_metric,
                     'train_pg_loss': train_pg_loss_metric,
                     'train_loss': train_loss_metric, }
    eval_metrics = {'eval_rew': eval_reward_metric,
                    'eval_ent': eval_ent_metric,
                    'eval_value_loss': eval_val_loss_metric,
                    'eval_pg_loss': eval_pg_loss_metric,
                    'eval_loss': eval_loss_metric, }
    test_metrics = {'test_rew': test_reward_metric,
                    'test_ent': test_ent_metric,
                    'test_value_loss': test_val_loss_metric,
                    'test_pg_loss': test_pg_loss_metric,
                    'test_loss': test_loss_metric, }
    return train_metrics, eval_metrics, test_metrics


def hasher(*elems):
    data, names, secs = elems
    namehashes = tf.strings.to_hash_bucket_fast(names, 20000)
    return data, namehashes, secs
