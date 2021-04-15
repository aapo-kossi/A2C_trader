# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:12:21 2020

@author: Aapo Kössi
"""

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# from tensorflow_probability.python.distributions import MultivariateNormalTriL as MVN        


#runner class and helper functions sf01 and get_discounted_rewards modified from OpenAI baselines A2C implementation at https://github.com/openai/baselines/blob/tf2/baselines/a2c/
class Runner:
    def __init__(self, env, model, nsteps, gamma, training = True):
        self.training = training
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma
        self.obs = env.current_time_step()
        self.obs_shape = env.obs_shape
        self.action_space = (env.num_envs, ) + env.action_space
        
        
    def run(self, until_done = False):
        mb_obs, mb_rewards, mb_actions, mb_raw_actions, mb_values, mb_dones, mb_mu, mb_L = [],[],[],[],[],[],[],[]
        
        def single_step():
            obs = self.obs
            actions, raw_actions, values, neg_cash_pen, neg_shares_pen, mu, L = self.model.step(obs, training = self.training)
            # Append the experiences
            mb_obs.append(obs.copy())
            mb_actions.append(actions)
            mb_raw_actions.append(raw_actions)
            mb_values.append(values)

            #arrange penalties (not currently utilized)
            neg_cash_pen = neg_cash_pen.numpy()
            neg_shares_pen = neg_shares_pen.numpy()            
            penalties = np.stack([neg_cash_pen, neg_shares_pen])
            
            mb_mu.append(mu)
            mb_L.append(L)

            # Take actions in env and record the results
            self.obs, rewards, dones = self.env.step(actions, penalties)
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            return

        if until_done:
            done = False
            while not done:
                single_step()
                done = mb_dones[-1][0]
        else:
            for n in range(self.nsteps):
                single_step()

        # batch of lists of steps to list of batches
        mb_obs = list(zip(*mb_obs))
        # list of batches of steps to list of batches of rollouts       
        mb_obs = [sf01(mb_obs[i]) for i in range(len(self.obs_shape))]
        mb_actions = sf01(mb_actions)
        mb_raw_actions = sf01(mb_raw_actions)
        mb_values = tf.transpose(mb_values, (1,0,2))
        
        mb_mu = sf01(mb_mu)
        mb_L = sf01(mb_L)
        

        if self.gamma > 0.0 and self.training:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs)
            last_values = tf.expand_dims(tf.squeeze(last_values, axis=-1), 0)
            done_in_end = mb_dones[-1] == [False]
            bstrap_rewards = tf.concat((mb_rewards, last_values), 0 )
            app = tf.fill(mb_dones[0].shape, False)
            bstrap_dones = mb_dones.copy()
            bstrap_dones.append(app)
            rewards_not_done = get_discounted_rewards(bstrap_rewards, bstrap_dones, self.gamma)[:-1]
            rewards_done = get_discounted_rewards(mb_rewards, mb_dones, self.gamma)
            mb_rewards = tf.where(done_in_end, rewards_done, rewards_not_done)
            
            
        mb_rewards = tf.transpose(tf.stack(mb_rewards), (1,0))
        mb_rewards = tf.reshape(mb_rewards, [-1])
        mb_values = tf.reshape(mb_values, [-1])
        return mb_obs, mb_rewards, mb_actions, mb_raw_actions, mb_values, mb_mu, mb_L
    

# TODO: somehow enable graph execution, issue being *s[2] iterating over a tensor
# @tf.function(experimental_relax_shapes=True)
def sf01(tensors):
    """
    stack tensors, swap and then flatten axes 0 and 1
    """
    tensor = tf.stack(tensors)
    s = tf.shape(tensor)
    rank = tf.rank(tensor)
    to_swap = tf.constant([1,0])
    if rank > 2:  
        trailing = tf.range(rank - 2) + 2
        perm = tf.concat((to_swap, trailing), 0)
    elif rank == 2:
        perm = to_swap
    else:
        tf.print('INVALID INPUT TENSOR FOR SF01 OP')
        perm = tf.constant([0])
    return tf.reshape(tf.transpose(tensor, perm), (s[0] * s[1], *s[2:]))

@tf.function(experimental_relax_shapes=True)
def get_discounted_rewards(rewards, dones, gamma):
    gamma = tf.cast(gamma, tf.float64)
    rewards = tf.cast(tf.stack(rewards), tf.float64)
    dones = tf.cast(tf.stack(dones), tf.float64)
    initializer = tf.zeros(rewards.shape[1], dtype = tf.float64)
    discounted = tf.scan(
        lambda acc, rew_done: rew_done[0] + gamma * acc * (tf.constant((1.,), dtype=tf.float64) - rew_done[1]),
        (rewards, dones),
        initializer,
        reverse = True)
    return discounted


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    