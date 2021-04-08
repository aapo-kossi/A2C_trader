# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:53:48 2020

@author: Aapo Kössi
"""
import time
import os.path as osp

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts as Cdr, InverseTimeDecay as Itd


# from tensorflow_probability.python.distributions import MultivariateNormalTriL as MVN

from vec_states import Runner

#classes A2CModel and the learn function slightly modified from baselines A2C implementation
#found at https://github.com/openai/baselines/blob/tf2/baselines/a2c/
class A2CModel:
    def __init__(self, env, model, lr_func, value_c, ent_c, max_grad_norm, alpha, epsilon):
        self.model = model
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr_func, rho = alpha, epsilon = epsilon)
        self.value_c = value_c
        self.ent_c = ent_c
        self.max_grad_norm = max_grad_norm
        
    @tf.function
    def step(self, obs, **kwargs):
        return self.model(obs, **kwargs)
    
    @tf.function
    def value(self, obs):
        return self.model.value(obs)

    @tf.function
    def train(self, obs, rewards, raw_actions, values, orig_mu, orig_L):
        advs = rewards - values
        
        #normalizing advantages:
        # advs = advs - tf.reduce_mean(advs) / (tf.keras.backend.std(advs) + 1e-8)
        
        with tf.GradientTape() as tape:
            mu, L, vpred = self.model(obs, dist_features = True)
            # tf.print(tf.round(mu), summarize = tf.size(mu))
            # tf.print(tf.round(orig_mu), summarize = tf.size(mu))
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
    
    
   
def learn(
    network,
    env,
    val_env = None,
    steps_per_update=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    init_lr=7e-2,
    decay_steps = 10000,
    decay_rate = 1e+1,
    t_mul = None,
    m_mul = None,
    lr_alpha = 0.0,
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=1e3,
    ckpt_interval = 1e5,
    val_interval = 1e3,
    load_path=None,
    MAR=None,
    **network_kwargs):
    
    nenvs = env.num_envs
   
    
    
    #instantiating the A2c model object
    #TODO: add actual network hparams
    if t_mul is not None and m_mul is not None:    
        lr_func = Cdr(init_lr, decay_steps, t_mul = t_mul, m_mul = m_mul, alpha = lr_alpha)
    else:
        lr_func = Itd(init_lr, total_timesteps, decay_rate)

    model = A2CModel(env, network, lr_func, vf_coef, ent_coef, max_grad_norm, alpha, epsilon)
    
    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model = model)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep = None)
        ckpt.restore(manager)
    
    runner = Runner(env, model, steps_per_update, gamma)
    
    val_runner = Runner(val_env, model, None, gamma, training = False)
    t_start = time.time()
    
    n_updates = total_timesteps // (nenvs * steps_per_update)
    fig, ax = plt.subplots()
    for update in range(1, n_updates + 1):
        
        obs, rewards, actions, raw_actions, values, mus, Ls = runner.run()
        policy_loss, value_loss, entropy, n_corrupt = model.train(obs, rewards, raw_actions, values, mus, Ls)
        if n_corrupt > 0:
            print(f'{n_corrupt} action(s) corrupt, for which no gradients propagated')
        nseconds = time.time() - t_start
        fps = int((update * nenvs * steps_per_update)/nseconds)

        if update % log_interval == 0:
            print("current single env steps fps: {}".format(fps))
            print("policy loss {:.3f},\n critic loss {:.3f},\n entropy loss {:.3f}".format(policy_loss, value_loss, entropy))
            print(f'mean reward for minibatch {update}: {tf.reduce_mean(rewards):.3g}')
            #TODO: add logging method for tensorboard
        if update % ckpt_interval == 0:
            print('Saving model not implemented yet!!')
            #TODO implement model weight saving
            raise SystemExit
            network.save_weights()
        if update % val_interval == 0:
            #TODO: add other metrics
            #TODO: refactor to enable loss calculation
            obs , rewards, _ , _ , _ ,_,_= val_runner.run(until_done=True, pure_profit = True)
            total_rewards = tf.reduce_sum(rewards)
            print(f'at update {update}, validation trajectory total rewards {total_rewards:.6f}. ')
            closes = obs[3]
            y_rew = tf.cumsum(rewards).numpy() / obs[4][0,0] + 1.0 
            x = range(rewards.shape[0])
            ax.clear()
            ax.plot(x, y_rew, 'r--')
            for i in range(1,closes.shape[-1]):
                ticker_price = closes[:,i]
                ax.plot(x, ticker_price / ticker_price[0], linewidth = 1)
            plt.show()
            plt.pause(0.01)
            
    return model


def neglogp(action, mu, L):
    n = tf.cast(action.shape[-1], tf.float64)

    vec_diff = tf.expand_dims(action - mu, -1)
    # tf.debugging.assert_equal(L, tf.eye(L.shape[1],batch_shape=[L.shape[0]],dtype=tf.float64), 'L not I')
    tf.debugging.assert_positive(tf.linalg.diag_part(L), 'L diagonal not positive')
    
    y = tf.linalg.triangular_solve(L, vec_diff)
    vec_diff_ = tf.linalg.matvec(L, tf.squeeze(y, axis = -1))
    good_L = tf.reduce_all(tf.abs(tf.squeeze(vec_diff) - vec_diff_) < 1e-5)

    ok_L = tf.reduce_all(tf.abs(tf.squeeze(vec_diff) - vec_diff_) < 1e-2)
    if not ok_L:
        tf.print('severe numerical issues')
    elif not good_L:
        tf.print('numerical issues')
    any_good_L = tf.reduce_any(tf.abs(tf.squeeze(vec_diff) - vec_diff_) < 1e-8)
    if not any_good_L:
        tf.print('network has gone to shit, no valid covariance matrices so cannot update weights')
    
    x = tf.linalg.triangular_solve(tf.linalg.matrix_transpose(L), y, lower = False)
    
    diffs_to_scale = 0.5 * tf.linalg.matrix_transpose(vec_diff) @ x
    # print(diffs_to_scale.shape)
    # tf.debugging.assert_all_finite(diffs_to_scale, 'diffs not finite')
    const = 0.5 * n * tf.math.log(tf.constant(2.0, dtype = tf.float64) * np.pi)
    # print(const.shape)
    # tf.debugging.assert_all_finite(const, 'const not finite')
    scale = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis = -1)
    # print(scale.shape)
    tf.debugging.assert_all_finite(scale, 'scale not finite')
    neglogp = const + scale + tf.squeeze(diffs_to_scale)
    # tf.print(tf.cast(neglogp,tf.int32))
    # tf.debugging.assert_all_finite(neglogp, 'what')
    n_corrupt = tf.reduce_sum(tf.cast(neglogp > 256,tf.int32))
    neglogp = tf.clip_by_value(neglogp, - 2.0 ** 9, 2.0 ** 8)
    return neglogp, n_corrupt

def entropy(mu, L):
    n = tf.cast(mu.shape[-1], tf.float64)
    ent = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis = -1) + \
          0.5 * n * (1.0 + tf.math.log(tf.constant(2, dtype = tf.float64) * np.pi))
    # print(ent)
    tf.debugging.assert_all_finite(ent, 'the hell??')
    return ent



