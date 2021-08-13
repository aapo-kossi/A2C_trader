# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:14:53 2020

@author: Aapo Kössi
"""

import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag as N
import constants

class TradingEnv:
    def __init__(self,
                 train_windows,
                 data_index,
                 sector_cats,
                 action_space,
                 n_envs = constants.N_ENVS,
                 init_capital = constants.STARTING_CAPITAL,
                 noise_ratio = 0.002,
                 vol_noise_intensity = 10,
                 cost_per_share = 0.0,
                 nec_penalty = 0.0,
                 nes_penalty = 0.0,
                 MAR = None,
                 render = False):
        
        self.init_capital = tf.cast(init_capital, tf.float64)
        self.window_data_index = tf.constant(data_index.to_numpy())
        self.data_index = drop_col(self.window_data_index, constants.others['data_index'].get_loc('date'))
        self.sector_cats = tf.constant(sector_cats)
        self.n_secs = self.sector_cats.shape[0]
        self.total_days = train_windows.element_spec[0].shape[0]
        self.input_days = constants.INPUT_DAYS
        self.action_space = action_space
        self.noisy = noise_ratio > 0.0
        self.noise_ratio = noise_ratio
        self.vol_noise_intensity = vol_noise_intensity
        self.MAR = MAR
        self.render = render
        self.pen_coef = tf.constant([nec_penalty, nes_penalty])
        self.cost_per_share = tf.constant(cost_per_share, dtype = tf.float64)
        
        self.num_envs = n_envs
        self.n_symbols = train_windows.element_spec[0].shape[1]
        self.window = iter(train_windows.prefetch(n_envs))

        self.capital = tf.Variable(tf.ones((n_envs, 1), dtype=tf.float64) * init_capital)
        self.equity = tf.Variable(tf.zeros([n_envs, self.n_symbols], dtype = tf.float64))
        self.returns = tf.Variable(tf.zeros(n_envs))
        self.n_step = tf.Variable(tf.zeros(n_envs, dtype = tf.int32))
        self.day = tf.Variable(self.n_step + self.input_days)
        self.one = tf.ones_like(self.n_step)
        
        init_ohlcvd = tf.zeros((n_envs,) + self.window.element_spec[0].shape[:-1] + (self.window.element_spec[0].shape[-1] - 1,))
        self.ohlcvd = tf.Variable(initial_value=init_ohlcvd)
        init_conames = tf.fill((self.num_envs,self.n_symbols), '')
        self.conames = tf.Variable(initial_value=init_conames)
        init_onehot_sectors = tf.zeros((self.num_envs, self.n_secs - 1, self.n_symbols))
        self.onehot_secs = tf.Variable(initial_value=init_onehot_sectors)
        init_dates = tf.zeros((self.num_envs, self.total_days))
        self.dates = tf.Variable(initial_value = init_dates)
        self.reset()
        self.obs_shape = [x.shape for x in self.current_time_step()]
        print('initialized environment')
    
    
    def current_time_step(self):
        """
        return obs as a list of tensors:
            0: companies, sectors, industries onehot encoded
            1: current equity
            2: historical ohlcvd data shaped (n_envs, n_tickers, n_days, n_channels)
            3: last prices
            4: current capital
        """
        onehots = tf.cast(self.onehot_secs, tf.float64)
        equity = self.equity
        ohlcvd = tf.stack([self.ohlcvd[i,self.day[i] - self.input_days:self.day[i]] for i in range(self.num_envs)])
        ohlcvd = tf.cast(tf.transpose(ohlcvd, perm = [0,2,1,3]), tf.float64)
        lasts = self.get_lasts()
        capital = self.capital
        # tf.print(tf.math.reduce_any(tf.math.is_nan(onehots)))
        # tf.print(tf.math.reduce_any(tf.math.is_nan(equity)))
        # tf.print(tf.math.reduce_any(tf.math.is_nan(ohlcvd)))
        # tf.print(tf.math.reduce_any(tf.math.is_nan(lasts)))
        # tf.print(tf.math.reduce_any(tf.math.is_nan(capital)))
        return [ tf.convert_to_tensor(ob, dtype=tf.float64) for ob in [onehots, equity, ohlcvd, lasts, capital]]
    
    @tf.function
    def reset(self):
        trues = tf.fill([self.num_envs], True)
        self._reset(trues)
        return
    
    def _reset(self, dones):
        index = tf.where(dones)
        n_dones = tf.size(index)
        # tf.print(f'resetting {n_dones} envs')
        new_ohlcvd, new_conames, new_secs = tf.map_fn(lambda i: next(self.window), index, parallel_iterations=8, fn_output_signature=(tf.float32, tf.string, tf.float32))            
        self.conames.scatter_nd_update(index, new_conames)
        sec_index = tf.cast(new_secs / 5 - 1, tf.int32)  #mapping from GICS sector (0,10,15,20... to index -1,1,2,3...)
        new_secs = tf.one_hot(sec_index, self.n_secs, axis = 1)[:,1:,:] #drop the first column as it is always empty (corresponds to GICS sector 5, which doesn't exist)
        new_dates = new_ohlcvd[:,:,0,tf.squeeze(tf.where(self.window_data_index == 'date'))]
        self.dates.scatter_nd_update(index, new_dates)
        self.onehot_secs.scatter_nd_update(index, new_secs)
        new_ohlcvd = drop_col(new_ohlcvd, tf.squeeze(tf.where(self.window_data_index == 'date')))
        if self.noisy:
            new_ohlcvd = self.add_noise(new_ohlcvd)
        self.ohlcvd.scatter_nd_update(index, new_ohlcvd)
        new_capital = tf.fill([n_dones,1], self.init_capital)
        self.capital.scatter_nd_update(index, new_capital)
        new_equity = tf.zeros([n_dones, self.n_symbols], dtype = tf.float64)
        self.equity.scatter_nd_update(index, new_equity)
        new_step = tf.zeros([n_dones], dtype = tf.int32)
        self.n_step.scatter_nd_update(index, new_step)
        new_day = new_step + self.input_days
        self.day.scatter_nd_update(index, new_day)
        # self.advance_to_wday()
        return
        
    #faster performance step function
    @tf.function
    def step(self, action, penalties):
        tf.debugging.assert_all_finite(action, 'action not finite...')
        #TODO: penalties not implemented and not a priority
        orig_mkt_value = self.get_mkt_val()
        lasts = self.get_lasts()
        divs = self.get_div()

        a = round_toward_0(action)
        a = clip_selling(a, self.equity)
        e = self.equity
        tf.debugging.assert_non_negative(e + a, message = 'selling more than available')

        n_traded = tf.math.reduce_sum(tf.math.abs(a), axis = 1)
        commissions = self.cost_per_share * n_traded
        commissions = tf.expand_dims(commissions, 1)

        self.capital.assign_add(tf.reduce_sum(- a * lasts + e * divs, axis = 1, keepdims=True) - commissions, read_value=False)
        self.equity.assign_add(a, read_value=False)
        tf.debugging.assert_non_negative(self.equity, f'negative equity {self.equity}')

        self.day.assign_add(self.one)
        self.n_step.assign_add(self.one)

        dones = self.day >= self.total_days
        on_margin = tf.squeeze(self.capital < 0.0)
        to_reset = tf.math.logical_or(dones, on_margin)

        rewards = self.get_rewards(orig_mkt_value, on_margin)
        if tf.reduce_any(to_reset): self._reset(to_reset)
        return self.current_time_step(), rewards, to_reset
        
    def render(self):
        """
        
        NOT IMPLEMENTED

        """
        pass
    

    def close(self):
        print(f'closed environment {self.__name__}')
        del self
        return
    
    def get_rewards(self, last_mkt_val, on_margin):
        profit = self.get_mkt_val() - last_mkt_val
        returns = profit / last_mkt_val * 100
        rewards = returns - 3.0 * tf.cast(on_margin, tf.float64)
        return rewards
    
    # def advance_to_wday(self):
    #     def get_cond():
    #         not_finished = self.day < self.total_days
    #         return tf.logical_and(self.market_closed(), not_finished)
    #     while tf.reduce_any(get_cond()):
    #         self.day.assign_add(tf.where(get_cond(),1,0))
    #     return
        
    
    def get_mkt_val(self):
        return tf.squeeze(self.capital + tf.reduce_sum(tf.cast(self.equity, tf.float64) * self.get_lasts(), axis = 1, keepdims=True), axis = -1)
    
    def get_lasts(self):
        lasts_key = tf.constant('prccd')
        return self.get_current_val(lasts_key)
        
    def get_div(self):
        div_key = tf.constant('dist')
        return self.get_current_val(div_key)
    
    # def market_closed(self):
    #     vol_key = tf.constant('cshtrd')
    #     vols = self.get_current_val(vol_key)
    #     closed = tf.reduce_all(vols == 0.0, axis = 1)
    #     return closed
        
    def get_current_val(self, feature):
        # today =  tf.stack([self.ohlcvd[i,self.day[i] - 1] for i in range(self.num_envs)])
        today = tf.gather(self.ohlcvd, self.day - 1, batch_dims=1)
        index = tf.squeeze(tf.where(self.data_index == feature))
        todays_val = today[...,index]
        return tf.cast(todays_val, tf.float64)
    
    def add_noise(self, ohlcvd):
        # price_keys = ['open', 'high', 'low', 'close']
        # price_i = tf.squeeze(tf.stack([tf.where(self.data_index == key) for key in price_keys]), axis=1)
        # vol_i = tf.squeeze(tf.where(self.data_index == 'volume'))
        ratio = self.noise_ratio
        # vol_noise_intensity = self.vol_noise_intensity
        # arr_size = ohlcvd.shape[-1]
        # ta = tf.TensorArray(tf.float32, size = arr_size, dynamic_size =False,element_shape = arr_elem_shape)
        # means = tf.reduce_mean(ohlcvd, axis = 0)
        # def normal_noise(i, ratio):
        #     return tf.random.normal(arr_elem_shape, stddev = means[...,i] * ratio)

        # def add_gnoise_to_elem(elem):
        #     def true_fn():
        #         return tf.clip_by_value(tf.random.normal((),mean = elem, stddev=elem * ratio), 5e-3,tf.float32.max)
        #     def false_fn():
        #         return elem
        #     return tf.cond(elem > 0.0, true_fn, false_fn)
        
        mu = ohlcvd
        std = mu * ratio
        noisy = N(mu, scale_diag = std).sample()
        noisy = tf.where(noisy == 0.0, 0.0, tf.clip_by_value(noisy, 5e-4, tf.float32.max))

        # print(noisy.shape)
        # for i in range(arr_size):
        #     if i == vol_i:
        #         ta = ta.write(i, tf.where(ohlcvd[...,i] > 0 ,ohlcvd[...,i] + normal_noise(i, vol_noise_intensity), ohlcvd[...,i]))
        #     else:
        #         cond = tf.reduce_any(i == price_i)
        #         if cond:
        #             ta = ta.write(i, tf.clip_by_value(ohlcvd[...,i] + normal_noise(i, ratio),0.005,tf.float32.max))
        #         else: ta = ta.write(i, ohlcvd[...,i])
        # noisy_ohlcvd =  tf.transpose(ta.stack(), (1,2,0))
        return noisy
    
    
def drop_col(tensor, col):
    before = tensor[...,:col]
    after = tensor[...,col + 1:]
    dropped = tf.concat([before, after],-1)
    return dropped

def clip_selling(action, equity):
    return tf.where(action < 0, tf.math.maximum(action, - equity), action)

def round_toward_0(tensor):
    return tf.where(tensor < 0, tf.math.ceil(tensor), tf.math.floor(tensor))
    
    
    