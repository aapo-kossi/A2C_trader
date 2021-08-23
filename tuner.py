# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 00:08:43 2021

@author: Aapo Kössi
"""

from datetime import datetime
import numpy as np
import tensorflow as tf
from keras_tuner.engine.tuner import Tuner
from TradingModel import Trader
from gym_tf_env import TradingEnv
from a2c import learn
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts, InverseTimeDecay
import constants

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

def pad_to_max_days(l, elem, name, sec):
    true_days = tf.shape(elem)[0]
    paddings = [[0,l - true_days],[0,0]]
    return tf.pad(elem, paddings), name, sec

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

#TODO: consider reverting to iterating indefinitely
def finish_ds(ds, arrs, training = False, window_l = constants.WINDOW_LENGTH, n_envs = None, seed = None, hp = None):
    if not training:
        ds = ds.interleave(lambda *x: window(*x, window_len = window_l), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(128, seed = seed).take(n_envs).cache().repeat()
    else:
        ds = ds.interleave(lambda *x: window(*x, window_len = window_l), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(constants.EP_SHUFFLE_BUF)
        #ds = ds.take(8192).cache().repeat().shuffle(8192)
    return ds

def make_lr_func(hp):
    def make_exp_decay(init_lr, hp):
        decay_steps = hp.Float('exp_decay_steps', min_value = 1e4, max_value = 1e7, sampling = 'log')
        return ExponentialDecay(init_lr, decay_steps, 0.96)
    def make_constant_lr(init_lr, hp):
        return init_lr
    def make_it_decay(init_lr, hp):
        decay_steps = hp.Float('it_decay_steps', min_value = 1e4, max_value = 1e7, sampling = 'log')
        return InverseTimeDecay(init_lr, decay_steps, 1.0)
    def make_cos_restarts_decay(init_lr, hp):
        first_decay_steps = hp.Float('it_decay_steps', min_value = 1e4, max_value = 1e7, sampling = 'log')
        return CosineDecayRestarts(init_lr, first_decay_steps, m_mul = hp.Float('cos_decay_m_mul', min_value = 1.0, max_value = 2.0, default = constants.M_MUL))
    
    init_lr = hp.Float('init_lr', min_value = 1e-7, max_value = 1e-2, sampling = 'log', default = constants.INIT_LR)
    func_map = {'exp_decay': make_exp_decay, 'constant_lr': make_constant_lr, 'it_decay': make_it_decay, 'cos_restarts_decay': make_cos_restarts_decay}
    chosen_type = hp.Choice('lr_schedule_type', ['exp_decay', 'constant_lr', 'it_decay', 'cos_restarts_decay'], default = 'cos_restarts_decay')
    return func_map[chosen_type](init_lr, hp)


class MyTuner(Tuner):
    def __init__(self, *args, max_model_size=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 distribution_strategy=None,
                 directory=None,
                 project_name=None,
                 logger=None,
                 tuner_id=None,
                 overwrite=False,
                 **kwargs):
        super(MyTuner, self).__init__(*args, max_model_size=max_model_size,
                                      optimizer=optimizer,
                                      loss=loss,
                                      metrics=metrics,
                                      distribution_strategy=distribution_strategy,
                                      directory=directory,
                                      project_name=project_name,
                                      logger=logger,
                                      tuner_id=tuner_id,
                                      overwrite=overwrite)
        self.__dict__.update(kwargs)
    
    def run_trial(self, trial, datasets):
        hp = trial.hyperparameters
        
        
        output_shape = tf.constant((self.n_stocks), dtype = tf.int32) 
        if "tuner/trial_id" in hp:
            past_trial = self.oracle.get_trial(hp["tuner/trial_id"])
            model = self.load_model(past_trial)
            optimizer_weights = hp.get('optimizer_weights')
            log_dir = hp.get('logdir')
        else:
            model = self.hypermodel.build(hp)
            optimizer_weights = None
            log_dir = f'logs/trader/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            hp.Fixed('logdir', log_dir)

        writer = tf.summary.create_file_writer(log_dir)

        input_days = hp.get('input_days') #this hparam is generated by the temporal dnn in order to ensure valid layer structure
        print(input_days)
        cost_p = hp.Float('cost_p', min_value = 0.0, max_value = 0.005, sampling = 'linear', default = constants.MIN_P_COMMISSION)
        cost_per_share = hp.Float('cost_per_share', min_value = 1e-5, max_value = 0.1, sampling = 'log', default = constants.COST_PER_SHARE)
        max_steps_env = hp.Int('max_steps_env', min_value = 16, max_value = 64, step = 16)
        train_ds, eval_ds, test_ds = datasets
        
        n_batch = hp.Int('n_batch', min_value = 8, max_value = 32, step = 8)
        train_ds = finish_ds(train_ds, self.train_arrs, window_l = input_days + max_steps_env, training = True, hp = hp)
        print(train_ds.element_spec)
        train_env = TradingEnv(train_ds, self.data_index,
                               self.sec_cats, (output_shape,),
                               n_envs = n_batch,
                               init_capital = hp.Int('init_capital', min_value = 10000, max_value = 50000, step = 20000),
                               noise_ratio = hp.Float('noise_volume', min_value = 0.0, max_value = 0.004, sampling = 'linear', default = constants.NOISE_RATIO),
                               vol_noise_intensity = hp.Choice('vol_noise_volume', [0,1,5,10,50], default = constants.VOL_NOISE_INTENSITY),
                               cost_per_share = cost_per_share,
                               cost_percentage = cost_p,
                               input_days = input_days)

        #TODO: instantiate validation envs and model with hparams
        eval_ds = finish_ds(eval_ds, self.eval_arrs, window_l = input_days + constants.VAL_STEPS,
                             n_envs = constants.N_VAL_ENVS, seed = 0)
        eval_env = TradingEnv(eval_ds, self.data_index,
                              self.sec_cats, (output_shape,),
                              n_envs = constants.N_VAL_ENVS,
                              init_capital = 30000,
                              noise_ratio = 0.0,
                              cost_per_share = cost_per_share,
                              cost_percentage = cost_p,
                              input_days = input_days)
        
        test_ds = finish_ds(test_ds, self.test_arrs, window_l = input_days + constants.TEST_STEPS,
                             n_envs = constants.N_TEST_ENVS, seed = 1)        
        test_env = TradingEnv(test_ds, self.data_index,
                              self.sec_cats, (output_shape,),
                              n_envs = constants.N_VAL_ENVS,
                              init_capital = 30000,
                              noise_ratio = 0.0,
                              cost_per_share = cost_per_share,
                              cost_percentage = cost_p,
                              input_days = input_days)
        
        init_epoch = hp['tuner/initial_epoch']
        last_epoch = hp['tuner/epochs']
        
        lr = make_lr_func(hp)
        #TODO: call a2c.learn with hparams
        for epoch in range(init_epoch, last_epoch):
            self.on_epoch_begin(trial, model, epoch, logs = None)
            trained_model, epoch_fitness, new_optimizer_weights = learn(
                  model,
                  train_env,
                  lr,
                  val_env = eval_env,
                  test_env = test_env,
                  steps_per_update=hp.Int('steps_per_update', min_value = 4, max_value = 32, step = 4, default = constants.N_STEPS_UPDATE),
                  eval_steps=constants.VAL_TIME,
                  test_steps=constants.TEST_TIME,
                  total_timesteps = 50000,
                  gamma= hp.Float('gamma', min_value = 0.0, max_value = 1.0, sampling = 'linear', default = constants.GAMMA),
                  log_interval = 10,
                  val_interval = 50000,
                  optimizer_init = optimizer_weights,
                  metrics = self.summary_metrics
                  )
            hp.Fixed('optimizer_weights', new_optimizer_weights)
            print(f'trained epoch {epoch} on trial {hp["tuner/trial_id"]}')
            self.on_epoch_end(trial, model, epoch, logs = {'fitness': epoch_fitness, **self.summary_metrics.result()})
            with writer.as_default():
                [tf.summary.scalar(x, self.summary_metrics[x].result(), step = epoch) for x in self.summary_metrics]
            [self.summary_metrics[x].reset_states() for x in self.summary_metrics]
        
        #this is gonna be fucking awesome on colab GPU
        
        
        
        
        
        
        
        
        