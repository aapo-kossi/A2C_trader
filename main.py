# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:30:53 2020

@author: Aapo Kössi
"""

import math
import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from bs4 import BeautifulSoup
import requests
from matplotlib import pyplot as plt
from TrainData import TrainData
from gym_tf_env import TradingEnv
from records import Records, FileData
from a2c import learn
import TradingModel
import constants
import arg_parser
#accesses datasets, starts training pipeline, monitors progress






def split_ds(ds, n_tickers):
    if constants.TRAIN_TIME + constants.VAL_TIME + constants.TEST_TIME != 1.0:
        print(('Dataset split into training, validation and test data incorrectly, '
              'contamination or data loss probable.'))
    daily_ds = ds.batch(n_tickers, drop_remainder=True)
    
    train_ds = daily_ds.take(int(constants.TRAIN_TIME * constants.TOTAL_TIME))
    val_ds   = daily_ds.skip(int(constants.TRAIN_TIME * constants.TOTAL_TIME)-constants.INPUT_DAYS)
    val_ds   = val_ds.take(constants.INPUT_DAYS + int(constants.VAL_TIME * constants.TOTAL_TIME))
    test_ds  = daily_ds.skip(int(constants.TOTAL_TIME * (1 - constants.TEST_TIME)) - constants.INPUT_DAYS)
    test_ds  = test_ds.take(constants.INPUT_DAYS + int(constants.TEST_TIME * constants.TOTAL_TIME))
    return train_ds, val_ds, test_ds

def make_sliding_windows(ds, length):
    windows = ds.window(length, shift = constants.WINDOW_DIFF, drop_remainder = True)
    single_elem_windows = windows.map(lambda elem: 
                                      tf.data.experimental.get_single_element(
                                          elem.batch(length, drop_remainder = True)))
    return single_elem_windows.repeat()

def contains_special(string):
    bools = list(map(lambda char: char in constants.SPECIAL_CHARS, string))
    return any(bools)

def main():
    
    print('started')
    plt.ion()
    
    args = arg_parser.parser.parse_args()
    if args.datafile is not None:
        complete_data = Records.read_record(args.datafile)
    else:
        print('no datafile provided, exitting...')
        raise SystemExit
        
    #TODO: can't imagine next line follows any sort of best practices
    constants.add('data_index', complete_data.data_index)

    train_ds, val_ds, test_ds = split_ds(complete_data.dataset, complete_data.num_tickers)
    
    
    #organize the training dataset into shuffled windows
    train_ds = train_ds.apply(lambda x: make_sliding_windows(x, constants.WINDOW_LENGTH))
    train_ds = train_ds.shuffle(constants.TOTAL_TIME // constants.WINDOW_DIFF + 1)
    
    eval_steps = int(constants.VAL_TIME * constants.TOTAL_TIME)
    test_steps = eval_steps
    val_ds = val_ds.apply(lambda x: make_sliding_windows(x, constants.INPUT_DAYS + eval_steps))
    test_ds = test_ds.apply(lambda x: make_sliding_windows(x, constants.INPUT_DAYS + test_steps))

    reversed_ticker_dict = {float(value) : key for (key, value) in complete_data.ticker_dict.items()}
    i = 0
    
    
    #visualized train_windows of stock performances
    vis = False
    if vis:
        for episode in train_ds:
            state = TradingEnv(episode,
                          complete_data.data_index,
                          complete_data.onehot_cats,
                          constants.STARTING_CAPITAL,)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            x = range(constants.W_UP_TIME)
            ydata = state.data()
            for j in range(complete_data.num_tickers):
                wavelength = complete_data.num_tickers
                omega = 2*math.pi*j/wavelength
                r_val = 0.5*(1 + math.sin(omega))
                g_val = 0.5*(1 + math.sin(omega + 2/3*math.pi))
                b_val = 0.5*(1 + math.sin(omega + 4/3*math.pi))
                color = [r_val, g_val, b_val]
                y = ydata[j,:,0]
                ax.plot(x, y,
                        label = reversed_ticker_dict[j],
                        color = color)
                ax.legend(ncol = 10)
            i += 1
            if i == 3: break

    # initialize envs and model
    output_shape = tf.constant((complete_data.num_tickers,), dtype = tf.int32)   
    vec_trading_env = TradingEnv(train_ds, complete_data.data_index,
                                complete_data.onehot_cats, (output_shape,),
                                n_envs = constants.N_ENVS,
                                init_capital = 50000, MAR = constants.RF, noise_ratio=constants.NOISE_RATIO)
   
    val_env = TradingEnv(val_ds, complete_data.data_index, complete_data.onehot_cats,
                         (output_shape,), n_envs = 1, init_capital=50000, MAR = constants.RF, noise_ratio=constants.NOISE_RATIO)
    model = TradingModel.Trader(output_shape)

    #learn using a2c algorithm
    learn(model,
          vec_trading_env,
          val_env = val_env,
          steps_per_update=constants.N_STEPS_UPDATE,
          eval_steps=int(eval_steps * 255/365),
          test_steps=int(test_steps * 255/365),
          init_lr = constants.INIT_LR,
          decay_steps = constants.INIT_DECAY_STEPS,
          decay_rate= constants.DECAY_RATE,
          t_mul = constants.T_MUL,
          m_mul = constants.M_MUL,
          lr_alpha = constants.LR_ALPHA,
          gamma=constants.GAMMA)
    
    
if __name__ == '__main__':
    main()