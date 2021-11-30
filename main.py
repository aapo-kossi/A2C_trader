# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:30:53 2020

@author: Aapo Kössi
"""


import tensorflow as tf
import keras_tuner as kt
import argparse
from gym_tf_env import TradingEnv
from a2c import learn
from tuner import MyTuner
import utils
import TradingModel
import constants
import sys

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
tf.config.optimizer.set_jit(True)
tf.config.set_soft_device_placement(False)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(918278)

def main():
    print('started')
    
    parser = argparse.ArgumentParser(description='Train a neural network to trade n stocks concurrently, '\
                                                 'provided a .csv file of stock data.')

    # parser.add_argument('save_path', help = 'path to the folder used to save model weights as checkpoints', type = str)
    parser.add_argument('-d', '--input_dir', help = 'path to the dir including processed csv split into train, eval and test dirs', type=str)
    parser.add_argument('-v', '--verbose', help = 'use this flag if you want logs detailing training progress while running', action = 'store_true')

    args = parser.parse_args()

    verbose = args.verbose
    parentdir = args.input_dir
    data_index = utils.get_data_index(parentdir)

    
    train_arrs, eval_arrs, test_arrs = utils.load_ids(parentdir)
    arr_list = [train_arrs, eval_arrs, test_arrs]
    
    sec_cats = utils.get_cats(train_arrs['sector_list'])

    train_ds, eval_ds, test_ds = utils.load_processed_datasets(parentdir, arr_list, data_index.get_loc('date'))
    
    
    # start = time.time()
    # for n, elem in enumerate(train_ds):
    #     if time.time() - start > 60: break
    #     taken = time.time() - start
    #     print(f'current fps: {n / taken}', end = '\r')
    #     # tf.print(elem, summarize = -1)
    #     # break
    # print('')
    # print(f'elements fetched in a minute: {n}')
    # raise SystemExit

    # mock_env = TradingEnv(test_ds,
    #                 data_index,
    #                 sec_cats,
    #                 tf.constant((constants.DEFAULT_TICKERS,), dtype = tf.int32),
    #                 noise_ratio = 0.0)
     
    # vis(mock_env)
    
    metrics = utils.setup_metrics()
    
    hypermodel = TradingModel.HyperTrader(out_shape = constants.DEFAULT_TICKERS, close_idx = data_index.get_loc('prccd'))
    
    hp = kt.HyperParameters()
    hp.Fixed('cost_per_share', constants.COST_PER_SHARE)
    hp.Fixed('cost_p', constants.MIN_P_COST)
    hp.Fixed('cost_minimum', constants.MIN_COST)
    hp.Fixed('steps_per_update', 16)
    
    oracle = kt.oracles.Hyperband(kt.Objective('fitness','max'), max_epochs = 50, hyperparameters = hp)
    
        
    tuner = MyTuner(oracle, hypermodel, summary_metrics = metrics, project_name = 'trader_optimization', n_stocks = constants.DEFAULT_TICKERS,
                    sec_cats = sec_cats, train_arrs = train_arrs,
                    eval_arrs = eval_arrs, test_arrs = test_arrs,
                    # save_path = args.save_path,
                    data_index = data_index,
                    verbose = verbose,
                    tune_new_entries=False)
    
    tuner.search((train_ds, eval_ds, test_ds), verbose=verbose)
    
    best_hparams = tuner.get_best_hyperparameters()
    print(best_hparams)

    
if __name__ == '__main__':
    main()
