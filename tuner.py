# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 00:08:43 2021

@author: Aapo Kössi
"""

from datetime import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hparams_api
from keras_tuner.engine.tuner import Tuner
from keras_tuner.engine.tuner_utils import convert_hyperparams_to_hparams
from gym_tf_env import TradingEnv
from a2c import learn
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts, InverseTimeDecay
import constants
import shutil

try:
    import google.colab
    IN_COLAB = True
except: IN_COLAB = False

def make_lr_func(hp):
    def make_exp_decay(init_lr, decay_steps, hp):
        return ExponentialDecay(init_lr, decay_steps, 0.96)
    def make_constant_lr(init_lr, hp, _):
        return init_lr
    def make_it_decay(init_lr, decay_steps, hp):
        return InverseTimeDecay(init_lr, decay_steps, 1.0)
    def make_cos_restarts_decay(init_lr, decay_steps, hp):
        return CosineDecayRestarts(init_lr, decay_steps, m_mul = hp.Float('cos_decay_m_mul', min_value = 1.0, max_value = 2.0, default = constants.M_MUL))
    
    init_lr = hp.Float('init_lr', min_value = 5e-6, max_value = 1e-4, sampling = 'log', default = constants.INIT_LR)
    decay_steps = hp.Float('it_decay_steps', min_value = 5e4, max_value = 1e6, sampling = 'log', default = constants.INIT_DECAY_STEPS)
    func_map = {'exp_decay': make_exp_decay, 'constant_lr': make_constant_lr, 'it_decay': make_it_decay, 'cos_restarts_decay': make_cos_restarts_decay}
    chosen_type = hp.Choice('lr_schedule_type', ['exp_decay', 'constant_lr', 'it_decay', 'cos_restarts_decay'], default = 'cos_restarts_decay')
    return func_map[chosen_type](init_lr, decay_steps, hp)


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

    
    #credit to @bberlo in https://github.com/keras-team/keras-tuner/issues/175
    def on_trial_end(self, trial):
        """A hook called after each trial is run.
        # Arguments:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        if trial.get_state().get("status") == 'INVALID':
            self.oracle.end_trial(trial.trial_id, status = "INVALID")
        else: self.oracle.end_trial(trial.trial_id)

        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()
        if IN_COLAB:
            # update the log directories that can persist across multiple VM sessions
            shutil.copytree(self.project_name, '/content/drive/mydrive', dirs_exist_ok=True)
            shutil.copytree('logs', '/content/drive/mydrive', dirs_exist_ok=True)
            
    def run_trial(self, trial, datasets, verbose = False):
        hp = trial.hyperparameters
        
        
        output_shape = tf.constant((self.n_stocks), dtype = tf.int32)
        if "tuner/trial_id" in hp:
            past_trial = self.oracle.get_trial(hp["tuner/trial_id"])
            model = self.load_model(past_trial)
            log_dir = hp.get('logdir')
        else:
            try:
                model = self.hypermodel.build(hp)
            except Exception:
                trial.status = 'INVALID'
                return
            time = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f'logs/trader_stable/{time}'
            hp.Fixed('logdir', log_dir)

        writer = tf.summary.create_file_writer(log_dir)

        input_days = hp.get('input_days') #this hparam is generated by the temporal dnn in order to ensure valid layer structure
        cost_p = hp.Float('cost_p', min_value = 0.0, max_value = 0.005, sampling = 'linear', default = constants.MIN_P_COST)
        cost_per_share = hp.Float('cost_per_share', min_value = 1e-5, max_value = 0.1, sampling = 'log', default = constants.COST_PER_SHARE)
        cost_minimum = hp.Float('cost_minimum', min_value = 0.0, max_value = 5.0, default = constants.MIN_COST)
        train_ds, eval_ds, test_ds = datasets
        
        
        n_batch = hp.Fixed('batch_size', value = 128)
        train_env = TradingEnv(train_ds, self.data_index,
                               self.sec_cats, (output_shape,),
                               n_envs = n_batch,
                               init_capital = hp.Int('init_capital', min_value = 10000, max_value = 50000, step = 20000),
                               noise_ratio = hp.Float('noise_volume', min_value = 0.0, max_value = 0.004, sampling = 'linear', default = constants.NOISE_RATIO),
                               vol_noise_intensity = hp.Choice('vol_noise_volume', [0,1,5,10,50], default = constants.VOL_NOISE_INTENSITY),
                               cost_per_share = cost_per_share,
                               cost_percentage = cost_p,
                               cost_minimum= cost_minimum,
                               input_days = input_days)

        eval_env = TradingEnv(eval_ds, self.data_index,
                              self.sec_cats, (output_shape,),
                              n_envs = constants.N_VAL_ENVS,
                              init_capital = 30000,
                              noise_ratio = 0.0,
                              cost_per_share = cost_per_share,
                              cost_percentage = cost_p,
                              cost_minimum= cost_minimum,
                              input_days = input_days)
        
        test_env = TradingEnv(test_ds, self.data_index,
                              self.sec_cats, (output_shape,),
                              n_envs = constants.N_VAL_ENVS,
                              init_capital = 30000,
                              noise_ratio = 0.0,
                              cost_per_share = cost_per_share,
                              cost_percentage = cost_p,
                              cost_minimum= cost_minimum,
                              input_days = input_days)
        
        steps_per_epoch = 5000000
        steps_per_update = hp.Int('steps_per_update', min_value = 8, max_value = 32, step = 8, default = constants.N_STEPS_UPDATE)
        updates_per_epoch = steps_per_epoch // (steps_per_update * n_batch) + 1
        init_epoch = hp['tuner/initial_epoch']
        last_epoch = hp['tuner/epochs']
        
        lr = make_lr_func(hp)
        epoch_logs = []
        for epoch in range(init_epoch, last_epoch):
            init_step = epoch * updates_per_epoch + 1
            trained_model, epoch_fitness = learn(
                  model,
                  train_env,
                  lr,
                  val_env = eval_env,
                  test_env = test_env,
                  steps_per_update=steps_per_update,
                  eval_steps=constants.VAL_TIME,
                  test_steps=constants.TEST_TIME,
                  total_timesteps = steps_per_epoch,
                  init_step = init_step,
                  vf_coef = hp.Float('vf_coef', min_value = 0.2, max_value = 5.0, sampling = 'log', default = 1.0),
                  gamma= hp.Float('gamma', min_value = 0.0, max_value = 1.0, sampling = 'linear', default = constants.GAMMA),
                  log_interval = 100,
                  val_iterations = 1,
                  metrics = self.summary_metrics,
                  writer = writer,
                  logdir = log_dir,
                  verbose = self.verbose
                  )
            
            logs = {'fitness': epoch_fitness}
            epoch_logs.append(logs)
            hparams = convert_hyperparams_to_hparams(hp)
            
            with writer.as_default():
                tf.summary.scalar('fitness', epoch_fitness, step = epoch)
                hparams_api.hparams(hparams)

        return logs
