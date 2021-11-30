# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:53:48 2020

@author: Aapo Kössi
"""
import time
import io
import os.path as osp
import constants

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts as Cdr, InverseTimeDecay as Itd


# from tensorflow_probability.python.distributions import MultivariateNormalTriL as MVN

from runner import Runner

#classes A2CModel and the learn function slightly modified from baselines A2C implementation
#found at https://github.com/openai/baselines/blob/tf2/baselines/a2c/
class A2CModel:
    def __init__(self, env, model, lr_func, value_c, ent_c, max_grad_norm, alpha, epsilon):
        self.model = model
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr_func, rho = alpha, epsilon = epsilon)
        self.value_c = value_c
        self.ent_c = ent_c
        self.max_grad_norm = max_grad_norm
        
    #already wrapped by calling Runner
    #@tf.function(jit_compile=True)
    def step(self, obs, **kwargs):
        return self.model(obs, **kwargs)
    
    #already wrapped by calling Runner
    #@tf.function(jit_compile=True)
    def value(self, obs):
        return self.model.value(obs)

    def train(self, obs, rewards, raw_actions, values, orig_mu, orig_L):
        advs = rewards - values
        #normalizing advantages:
        # advs = advs - tf.reduce_mean(advs) / (tf.keras.backend.std(advs) + 1e-8)
        
        with tf.GradientTape() as tape:
            mu, L, vpred = self.model(obs, dist_features = True)
            # tf.debugging.assert_near(L, orig_L, rtol=0.0001, atol = 0.0001)
            neglogpac, n_corrupt = neglogp(raw_actions, mu, L)
            entropy = self.entropy_loss(mu, L)
            vf_loss = self.value_loss(vpred, rewards)
            pg_loss = self.action_loss(neglogpac, advs)
            loss = pg_loss - entropy + vf_loss
        
        var_list = tape.watched_variables()
        grads = tape.gradient(loss, var_list)
        grads, norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        [tf.debugging.check_numerics(x, 'grads {} not finite'.format(n)) for n, x in enumerate(grads)]
        grads_and_vars = list(zip(grads, var_list))
        self.optimizer.apply_gradients(grads_and_vars)
        [tf.debugging.check_numerics(x, 'vars {} not finite'.format(n)) for n, x in enumerate(var_list)]
        # raise SystemExit
        return pg_loss, vf_loss, entropy, n_corrupt       

    def action_loss(self, neglogp, advantage):
        return tf.reduce_mean( neglogp * advantage )
        
    def value_loss(self, vpreds, rewards):
        return self.value_c * tf.reduce_mean(tf.square(rewards - vpreds))
    
    def entropy_loss(self, mu, L):
        entropies = entropy(mu, L)
        return tf.reduce_mean(entropies) * self.ent_c
    
def neglogp(action, mu, L):
    n = tf.cast(action.shape[-1], tf.float64)

    vec_diff = tf.expand_dims(action - mu, -1)

    y = tf.linalg.triangular_solve(L, vec_diff)
    # vec_diff_ = tf.linalg.matvec(L, tf.squeeze(y, axis = -1))
#    good_L = tf.reduce_all(tf.abs(tf.squeeze(vec_diff) - vec_diff_) < 1e-5)

 #   ok_L = tf.reduce_all(tf.abs(tf.squeeze(vec_diff) - vec_diff_) < 1e-2)
  #  if not ok_L:
   #     tf.print('severe numerical issues')
   # elif not good_L:
   #     tf.print('numerical issues')
   # any_good_L = tf.reduce_any(tf.abs(tf.squeeze(vec_diff) - vec_diff_) < 1e-8)
   # if not any_good_L:
   #     tf.print('no valid covariance matrices, updating weights impossible')
    
    x = tf.linalg.triangular_solve(tf.linalg.matrix_transpose(L), y, lower = False)
    
    diffs_to_scale = 0.5 * tf.linalg.matrix_transpose(vec_diff) @ x
    # print(diffs_to_scale.shape)
    # tf.debugging.assert_all_finite(diffs_to_scale, 'diffs not finite')
    const = 0.5 * n * tf.math.log(tf.constant(2.0, dtype = tf.float64) * np.pi)
    # print(const.shape)
    # tf.debugging.assert_all_finite(const, 'const not finite')
    scale = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis = -1)
    # print(scale.shape)
    #tf.debugging.assert_all_finite(scale, 'scale not finite')
    neglogp = const + scale + tf.squeeze(diffs_to_scale)
    # tf.debugging.assert_all_finite(neglogp, 'what')
    n_corrupt = tf.reduce_sum(tf.cast(neglogp > 256,tf.int32))
    neglogp = tf.clip_by_value(neglogp, - 2.0 ** 9, 2.0 ** 8)
    return neglogp, n_corrupt

def entropy(mu, L):
    n = tf.cast(mu.shape[-1], tf.float64)
    ent = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis = -1) + \
          0.5 * n * (1.0 + tf.math.log(tf.constant(2, dtype = tf.float64) * np.pi))
    # print(ent)
    #tf.debugging.assert_all_finite(ent, 'the hell??')
    return ent

def write_plot(fig, writer, step):
    buf = io.BytesIO()
    fig.savefig(buf, format = 'png', dpi = 200)
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels = 4)
    image = tf.expand_dims(image, 0)
    with writer.as_default():
        tf.summary.image("Validation Trajectories", image, step=step)
    return image
  
   
def learn(
    network,
    env,
    # ckpt_path,
    lr_func,
    # initial_ckpt = None,
    val_env = None,
    test_env = None,
    steps_per_update=5,
    eval_steps = 100,
    test_steps = 100,
    total_timesteps=int(80e7),
    init_step = 1,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=50,
    ckpt_interval = 1e4,
    val_iterations = 1,
    MAR=None,
    metrics = ({},{},{}),
    writer = None,
    logdir=None,
    verbose = False,
    **network_kwargs):
    
    nenvs = env.num_envs
    train_metrics, eval_metrics, test_metrics = metrics
    #instantiating the A2c model, set correct optimizer step
    model = A2CModel(env, network, lr_func, vf_coef, ent_coef, max_grad_norm, alpha, epsilon)
    set_optimizer_iter(env, model, model.optimizer, init_step-1)

    #instantiate runners
    runner = Runner(env, model, steps_per_update, gamma)
    val_runner = Runner(val_env, model, eval_steps, 0.0)
    
    
    @tf.function#(jit_compile=True)
    def train_batch():
        obs, rewards, actions, raw_actions, values, mus, Ls = runner.run()
        policy_loss, value_loss, entropy, n_corrupt = model.train(obs, rewards, raw_actions, values, mus, Ls)
        return policy_loss, value_loss, entropy, n_corrupt, rewards

    def validate(update):
        xplots = np.ceil(np.sqrt(val_env.num_envs)).astype(np.int32)
        yplots = np.ceil(val_env.num_envs / xplots).astype(np.int32)
        fig, axs = plt.subplots(xplots, yplots)
        obs , rewards, actions, raw_actions, values, mus, Ls = val_runner.val_run(until_done=True, bootstrap=False)
        total_rewards = tf.reduce_sum(rewards) / tf.cast(tf.size(rewards), tf.float64)
        neglogpac, _ = neglogp(raw_actions, mus, Ls)
        advs = rewards - values
        pg_loss = model.action_loss(neglogpac, advs)
        value_loss = model.value_loss(values, rewards)
        entropy_loss = model.entropy_loss(mus, Ls)
        eval_metrics['eval_loss'].update_state(pg_loss + value_loss + entropy_loss)
        eval_metrics['eval_pg_loss'].update_state(pg_loss)
        eval_metrics['eval_value_loss'].update_state(value_loss)
        eval_metrics['eval_ent'].update_state(entropy_loss)
        eval_metrics['eval_rew'].update_state(rewards)
        closes = obs[3]
        split_closes = tf.split(closes, num_or_size_splits = val_env.num_envs)
        split_rewards = tf.split(rewards, num_or_size_splits = val_env.num_envs)
        for n in range(val_env.num_envs):
            ax_x = n % xplots
            ax_y = n // yplots
            y_rew = tf.math.cumprod(split_rewards[n] / 100 + 1.0)
            x = range(split_rewards[n].shape[0])
            axs[ax_x, ax_y].clear()
            axs[ax_x, ax_y].plot(x, y_rew, 'r--')
            for i in range(1,closes.shape[-1]):
                ticker_price = split_closes[n][...,i]
                axs[ax_x, ax_y].plot(x, ticker_price / ticker_price[0], linewidth = 1, alpha = 0.5)
        if writer is not None:
            write_plot(fig, writer, update)
            with writer.as_default():
                [tf.summary.scalar(x, eval_metrics[x].result(), step = update) for x in eval_metrics]
            [eval_metrics[x].reset_states() for x in eval_metrics]
        else:
            plt.pause(0.3)
            print(f'at update {update}, validation trajectory average daily rewards {total_rewards:.6f}. ')
            print(f'validation losses:\n     value loss {value_loss:.3f}\n     policy loss {pg_loss:.3f}\n     entropy loss {entropy_loss:.3f}')
        val_env.reset()
        return tf.reduce_sum(rewards)
    
    
    t_start = time.time()
    n_updates = total_timesteps // (nenvs * steps_per_update) + 1
    if val_iterations > 0:
        val_updates = n_updates // val_iterations
    else: val_updates = 0
    accumulated_reward = 0.0


    for update in range(init_step, init_step + n_updates):
        if update == 11:
            with tf.profiler.experimental.Trace('train', step_num = update):
                policy_loss, value_loss, entropy, n_corrupt, rewards = train_batch()
        elif update == 1 and writer is not None:
            tf.summary.trace_on(graph = True)
            policy_loss, value_loss, entropy, n_corrupt, rewards = train_batch()
            with writer.as_default():
                tf.summary.trace_export(name= "train_trace", step = 0,profiler_outdir=logdir)
        else:
            policy_loss, value_loss, entropy, n_corrupt, rewards = train_batch()    

        train_metrics['train_loss'].update_state(policy_loss + value_loss + entropy)
        train_metrics['train_pg_loss'].update_state(policy_loss)
        train_metrics['train_value_loss'].update_state(value_loss)
        train_metrics['train_ent'].update_state(entropy)
        train_metrics['train_rew'].update_state(rewards)
        
        #for first update, print nn summaries
        if update == 1:
            model.model.summary()
            model.model.get_layer('temporal_network').summary()
            
        #possibly print information about numerically unstable steps
        if n_corrupt > 0:
            print(f'{n_corrupt} action(s) corrupt, for which no gradients propagated')
        nseconds = time.time() - t_start
        fps = int(((update - init_step) * nenvs * steps_per_update)/nseconds)

        #intermittently write to logs/print training progress
        if  log_interval != 0 and update % log_interval == 0:
            if verbose:
                print("current single env steps fps: {}".format(fps))
                print("     policy loss {:.3f},\n     critic loss {:.3f},\n entropy loss {:.3f}".format(policy_loss, value_loss, entropy))
                print(f'mean reward for minibatch {update}: {tf.reduce_mean(rewards):.3g}')
            if writer is not None:
                with writer.as_default():
                    [tf.summary.scalar(x, train_metrics[x].result(), step = update) for x in train_metrics]
                [train_metrics[x].reset_state() for x in train_metrics]

        #profile the 11th training step
        if update == 10:
            tf.profiler.experimental.start('logs/trader')
        if update == 11:
            tf.profiler.experimental.stop('logs/trader')
            
        #evaluate model fitness
        if val_updates != 0 and update % val_updates == 0 and val_env is not None:
            accumulated_reward += validate(update)
                        
    return model, accumulated_reward

def set_optimizer_iter(env, model, optimizer, step):
    obs = env.current_time_step()
    model.step(obs)
    trainable_vars = model.model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in trainable_vars]
    model.optimizer.apply_gradients(zip(zero_grads, trainable_vars))
    opt_vars = model.optimizer.get_weights()
    opt_vars[0] = np.array(step)
    model.optimizer.set_weights(opt_vars)






