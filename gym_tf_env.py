# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:14:53 2020

@author: Aapo Kössi
"""

import tensorflow as tf
import constants

class TradingEnv:
    def __init__(self,
                 train_windows,
                 data_index,
                 onehots,
                 action_space,
                 n_envs = 16,
                 init_capital = 50000,
                 noisy = True,
                 noise_ratio = 0.002,
                 vol_noise_intensity = 10,
                 nec_penalty = 0.0,
                 nes_penalty = 0.0,
                 MAR = None,
                 render = False):
        
        self.init_capital = tf.cast(init_capital, tf.float32)
        self.data_index = tf.constant(data_index.to_numpy())
        self.onehots = tf.repeat(tf.expand_dims(tf.constant(onehots.to_numpy()),0), n_envs, axis=0)
        self.total_days = train_windows.element_spec.shape[0]
        self.input_days = constants.INPUT_DAYS
        self.action_space = action_space
        self.noisy = noisy
        self.noise_ratio = noise_ratio
        self.vol_noise_intensity = vol_noise_intensity
        self.MAR = MAR
        self.render = render
        self.pen_coef = tf.constant([nec_penalty, nes_penalty])
        
        self.num_envs = n_envs
        self.n_symbols = train_windows.element_spec.shape[1]
        self.window = iter(train_windows.prefetch(n_envs))

        self.capital = tf.Variable(tf.ones((n_envs, 1)) * init_capital)
        self.equity = tf.Variable(tf.zeros([n_envs, self.n_symbols], dtype = tf.int32))
        self.returns = tf.Variable(tf.zeros(n_envs))
        self.n_step = tf.Variable(tf.zeros(n_envs, dtype = tf.int32))
        self.day = tf.Variable(self.n_step + self.input_days)
        
        init_ohlcvd = tf.stack([next(self.window) for _ in range(n_envs)])
        self.ohlcvd = tf.Variable(initial_value=init_ohlcvd)
        self.obs_shape = [x.shape for x in self.current_time_step()]
    
    
    def current_time_step(self):
        """
        return obs as a list of tensors:
            0: companies, sectors, industries onehot encoded
            1: current equity
            2: historical ohlcvd data shaped (n_envs, n_tickers, n_days, n_features)
            3: last prices
            4: current capital
        """
        onehots = self.onehots
        equity = tf.cast(self.equity, dtype = tf.float32)
        ohlcvd = tf.stack([self.ohlcvd[i,self.day[i] - self.input_days:self.day[i]] for i in range(self.num_envs)])
        ohlcvd = tf.transpose(ohlcvd, perm = [0,2,1,3])
        lasts = self.get_lasts()
        capital = self.capital
        return [onehots, equity, ohlcvd, lasts, capital]
    
    def reset(self):
        trues = tf.fill(self.num_envs, True)
        self._reset(trues)
        return
    
    def _reset(self, dones):
        index = tf.where(dones)
        n_dones = tf.size(index)
        tf.print(n_dones)
        new_ohlcvd = tf.map_fn(lambda _: self.add_noise(next(self.window)), index, fn_output_signature =tf.float32)
        self.ohlcvd.scatter_nd_update(index, new_ohlcvd)
        new_capital = tf.fill([n_dones,1], self.init_capital)
        self.capital.scatter_nd_update(index, new_capital)
        new_equity = tf.zeros([n_dones, self.n_symbols], dtype = tf.int32)
        self.equity.scatter_nd_update(index, new_equity)
        new_step = tf.zeros([n_dones], dtype = tf.int32)
        self.n_step.scatter_nd_update(index, new_step)
        new_day = new_step + self.input_days
        self.day.scatter_nd_update(index, new_day)
        self.advance_to_wday()
        return
        
    #faster performance step function
    @tf.function
    def step(self, action, penalties):
        #TODO: penalties not implemented and not a priority
        orig_mkt_value = self.get_mkt_val()
        lasts = self.get_lasts()
        divs = self.get_div()
        a = tf.cast(tf.cast(action, tf.int32),tf.float32)
        e = tf.cast(self.equity, tf.float32)
        self.capital.assign_add(tf.reduce_sum(- a * lasts + e * divs, axis = 1, keepdims=True), read_value=False)
        a = tf.cast(action, tf.int32)
        self.equity.assign_add(a, read_value=False)
        tf.debugging.assert_non_negative(self.equity, 'negative equity')

        self.day.assign_add(tf.ones_like(self.day))
        self.advance_to_wday()
            
        self.n_step.assign_add(tf.ones_like(self.n_step))
        dones = self.day >= self.total_days
        rewards = self.get_rewards(orig_mkt_value)
        self._reset(dones)
        return self.current_time_step(), rewards, dones
        
    def render(self):
        """
        
        NOT IMPLEMENTED

        """
        pass
    

    def close(self):
        del self
        return
    
    def get_rewards(self, last_mkt_val):
        profit = self.get_mkt_val() - last_mkt_val
        returns = profit / last_mkt_val * 100
        return returns
    
    def advance_to_wday(self):
        def get_cond():
            not_finished = self.day < self.total_days
            return tf.logical_and(self.market_closed(), not_finished)
        while tf.reduce_any(get_cond()):
            self.day.assign_add(tf.where(get_cond(),1,0))
        
    
    def get_mkt_val(self):
        return tf.squeeze(self.capital + tf.reduce_sum(tf.cast(self.equity, tf.float32) * self.get_lasts(), axis = 1, keepdims=True), axis = -1)
    
    def get_lasts(self):
        lasts_key = tf.constant('close')
        return self.get_current_val(lasts_key)
        
    def get_div(self):
        div_key = tf.constant('divCash')
        return self.get_current_val(div_key)
    
    def market_closed(self):
        vol_key = tf.constant('volume')
        vols = self.get_current_val(vol_key)
        closed = tf.reduce_all(vols == 0.0, axis = 1)
        return closed
        
    def get_current_val(self, feature):
        today =  tf.stack([self.ohlcvd[i,self.day[i] - 1] for i in range(self.num_envs)])
        index = tf.squeeze(tf.where(self.data_index == feature))
        todays_val = today[...,index]
        return todays_val
    
    def add_noise(self, ohlcvd):
        price_keys = ['open', 'high', 'low', 'close']
        price_i = tf.squeeze(tf.stack([tf.where(self.data_index == key) for key in price_keys]), axis=1)
        vol_i = tf.squeeze(tf.where(self.data_index == 'volume'))
        ratio = self.noise_ratio
        vol_noise_intensity = self.vol_noise_intensity
        arr_size = ohlcvd.shape[-1]
        arr_elem_shape = ohlcvd[...,0].shape
        ta = tf.TensorArray(tf.float32, size = arr_size, dynamic_size =False,element_shape = arr_elem_shape)
        means = tf.reduce_mean(ohlcvd, axis = 0)

        def normal_noise(i, ratio):
            return tf.random.normal(arr_elem_shape, stddev = means[...,i] * ratio)

        for i in range(arr_size):
            if i == vol_i:
                ta = ta.write(i, tf.where(ohlcvd[...,i] > 0 ,ohlcvd[...,i] + normal_noise(i, vol_noise_intensity), ohlcvd[...,i]))
            else:
                flag = False
                cond = tf.reduce_any(i == price_i)
                if cond:
                    ta = ta.write(i, ohlcvd[...,i] + normal_noise(i, ratio))
                    flag = True
                if not flag: ta = ta.write(i, ohlcvd[...,i])
        noisy_ohlcvd =  tf.transpose(ta.stack(), (1,2,0))
        return noisy_ohlcvd
    
    
    
    
    