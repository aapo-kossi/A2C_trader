# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:41:53 2020

@author: Aapo Kössi
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import constants

class VecStates:
    def __init__(self, train_windows,
                 data_index,
                 onehots,
                 action_space,
                 n_workers = 16,
                 init_capital = 50000,
                 MAR = None,
                 render = False):
        
        #TODO: implement rendering of portfolio performances 
        # (overall performance against individual stocks and m-cap weighted index)
        # and portfolio components over time (utilizing colours for different stocks)
        # only visualize first env from list, if batched
        
        self.action_space = action_space
        self.window_iterator = iter(train_windows.prefetch(n_workers))
        self.n_workers = n_workers
        self.num_envs = n_workers
        self.MAR = MAR
        self.data_index = data_index
        self.onehots = onehots
        self.init_capital = tf.constant((init_capital,), dtype = tf.float32)
        self.states = self.generate_states(self.n_workers)
        self()
        self.obs_shape = tuple(next(map(lambda x: x.obs_shape , self.states)))
        
        
    def __call__(self):
        obs = list(map(lambda x: x(), self.states))
        self.obs = [tf.stack(x) for x in zip(*obs)]
        return self.obs
    
        #generating list of States
    def generate_states(self, num):
        return np.array([State(next(self.window_iterator), self.data_index, self.onehots, self.init_capital, MAR = self.MAR) for _ in range(num)])

        
    def step(self, actions, penalties = None, **kwargs):        
        """
        1. take step according to actions-vector in the states-vector, return vector of what states return
        2. replace finished states with new ones
        TODO: @tf.function compatibility!!!!!!
        """
        dones = tf.TensorArray(tf.bool, size=0,dynamic_size=True)
        rewards = tf.TensorArray(tf.float32, size=0,dynamic_size=True)
        for n, state in enumerate(self.states):
            _ , reward, done = state.advance(actions[n], np.squeeze(penalties[:,n]), **kwargs)
            dones = dones.write(n, done)
            rewards = rewards.write(n, reward)
            
        rewards = rewards.stack()
        dones = dones.stack()
        #replace done environments with fresh ones
        n_dones = tf.reduce_sum(tf.cast(dones, tf.uint8))
        fresh = self.generate_states(n_dones)
        self.states[dones.numpy()] = fresh
        
        #get the next observations
        self.obs = self()
        return self.obs, rewards, dones


class State:
    #TODO: first input is an element of a dataset containing the same data
    #as the dataset previously did, fix everything accordingly
    """
    
    Handles operations related to a single episode of training the trading algorithm
    with the key method __call__    
    
    """
    #broken
    def __init__(self,
                 ohlcvd: tf.Tensor,
                 data_index: pd.Index,
                 onehot_cats: pd.core.frame.DataFrame,
                 capital,
                 noisy = True,
                 noise_ratio = 0.002,
                 vol_noise_intensity = 10,
                 nec_penalty = 0.0,
                 nes_penalty = 0.0,
                 MAR = None):
        
        if noisy:
            price_keys = ['close', 'high','low','open']
            self.__add_noise(ohlcvd,
                           price_i = [data_index.get_loc(key) for key in price_keys],
                           vol_i = data_index.get_loc('volume'),
                           ratio = noise_ratio,
                           vol_noise_intensity = vol_noise_intensity)

        self.total_days = ohlcvd.shape[0]
        self.__data_index = data_index
        self.__ohlcvd = ohlcvd
        self.onehot_cats = onehot_cats.astype(np.uint8).to_numpy()
        self.starting_capital = capital
        self.capital = capital
        self.day = constants.W_UP_TIME
        self.num_step = 0
        self.equity = np.zeros(self.__ohlcvd.shape[1], dtype = np.int32)
        self.profit = np.array([0.0])
        self.pen_c = np.array([nec_penalty, nes_penalty])
        self.obs_shape = self.__observation_shape()
        self.lasts, self.dividends = self.lasts_and_dividends()
        self.MAR = MAR

    def __call__(self):
        inputs = [self.onehot_cats, self.equity, self.data(), self.lasts_and_dividends()[0], self.capital]
        obs = [tf.cast(tf.constant(inputs[i]), tf.float32) for i in range(len(inputs))]
        return obs
        
    def __observation_shape(self):
        test_call = self()
        observation_shape = [test_call[i].shape for i in range(len(test_call))]
        return observation_shape
    
    #refactor for graph execution, self.profit causes retracing
    def get_rewards(self, penalties, pure_profit = False):
        penalty = sum(self.pen_c[penalties])
        if self.MAR is not None and not pure_profit:
            rew =  self.get_sortino(self.MAR) - penalty
        else:
            rew = self.profit[-1] - self.profit[-2] - penalty
        return tf.constant(rew, dtype = tf.float32)
            
    
    def get_sortino(self, MAR):
        
        daily_profit = self.profit[1:] - self.profit[:-1]
        daily_return = (daily_profit / self.starting_capital) * 100
        mean_return = np.mean(daily_return) * constants.N
        # print(mean_return)
        mask = daily_profit < MAR
        downside_return = daily_return[mask]
        down_std = np.std(downside_return) * np.sqrt(constants.N) # not a genetic sexually transmitted disease

        #return profit - MAR for first day, since downside deviation isn't defined yet
        if (not np.isnan(down_std)) and down_std != 0.0:
            reward =  (mean_return - MAR) / down_std
        else: reward = mean_return - MAR
        return reward
    
    def done(self):
        return self.day + 1 >= self.total_days
        
    def data(self):
        """
        Returns
        -------
        Tensor containing the historical data
        according to State.day.
        Length of history is determined by
        constants.W_UP_TIME.
        Returned tensor has shape(num_tickers, num_days, num_values)
        """
        
        ohlcvd_history = self.__ohlcvd[ self.day - constants.W_UP_TIME : self.day ]
        return tf.transpose(ohlcvd_history, perm = [ 1,0,2 ])
    
    def advance(self, action, penalties, **kwargs):
        """
        THE MAIN INTERFACE OF THE CLASS
        action: list(num_shares_to_buy, can be pos or neg)
        assumes legal actions
        advances to the next day in this window of the dataset
        updates state of the model according to the action input
        """


        lasts, dividends = self.lasts_and_dividends()
        action = np.array(action).astype(np.int32)
        #adjust amount of shares held:
        equity = self.equity + action
        self.equity = equity.astype(np.int32)
        # print(self.equity)
        #adjust capital according to decisions:
        self.capital -= np.sum(action * lasts)
        # print(self.capital)
        #add capital according to dividends:
        self.capital += np.sum(self.equity * dividends)
        mkt_open = False
        while not mkt_open and not self.done():
            self.day += 1
            mkt_open = self.market_open()
        self.profit = np.append(self.profit, self.market_value() - self.starting_capital)
        self.num_step += 1
        return self(), self.get_rewards(penalties, **kwargs), self.done()
            
    def lasts_and_dividends(self):
        """

        Returns
        -------
        np.array closes : Closing prices of all stocks on the current day of this object.
        np.array dividends : Dividends of all stocks paid on the current day of this object.

        """
        today = self.__ohlcvd[self.day]
        
        closes = today[:,self.__data_index.get_loc('close')]
        dividends = today[:,self.__data_index.get_loc('divCash')]
        return closes, dividends
        
        
    def market_value(self):
        lasts, dividends = self.lasts_and_dividends()
        return self.capital + np.sum(lasts * self.equity)
    
    def total_profit(self):
        return self.profit[-1]
    
    def market_open(self):
        today = self.__ohlcvd[self.day]
        vol = today[:, self.__data_index.get_loc('volume')]
        return tf.reduce_any(vol != 0.0)
        
    @staticmethod
    def __add_noise(ohlcvd, price_i = [], vol_i = None, ratio = 0.002, vol_noise_intensity = 0): 
        arr_size = ohlcvd.shape[-1]
        arr_elem_shape = ohlcvd[...,0].shape
        ta = tf.TensorArray(tf.float32, size = arr_size, dynamic_size =False,element_shape = arr_elem_shape)
        means = tf.reduce_mean(ohlcvd, axis = 0)

        def normal_noise(i, ratio):
            return tf.random.normal(arr_elem_shape, stddev = means[...,i] * ratio)

        for i in range(arr_size):
            if i == vol_i:
                ta = ta.write(i, ohlcvd[...,i] + normal_noise(i, vol_noise_intensity))
            else:
                flag = False
                for index in price_i:
                    if i == index:
                        ta = ta.write(i, ohlcvd[...,i] + normal_noise(i, ratio))
                        flag = True
            if not flag: ta = ta.write(i, ohlcvd[...,i])
        noisy_ohlcvd =  tf.transpose(ta.stack(), (1,2,0))
        return noisy_ohlcvd
                    
        
        
        
        
        
        
        