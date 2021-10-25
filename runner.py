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
        self.obs_shape = env.obs_shape
        self.action_space = (env.num_envs, ) + env.action_space
        
    @tf.function
    def run(self, until_done = False, bootstrap = True):
        last_obs = self.env.current_time_step()
        mb_obs = [tf.TensorArray(tf.float64, size=0, dynamic_size = True), tf.TensorArray(tf.float64, size=0, dynamic_size = True), tf.TensorArray(tf.float64, size=0, dynamic_size = True), tf.TensorArray(tf.float64, size=0, dynamic_size = True), tf.TensorArray(tf.float64, size=0, dynamic_size = True), ]
        mb_rewards = tf.TensorArray(tf.float64, size=0, dynamic_size = True)
        mb_actions = tf.TensorArray(tf.float64, size=0, dynamic_size = True)
        mb_raw_actions = tf.TensorArray(tf.float64, size=0, dynamic_size = True)
        mb_values = tf.TensorArray(tf.float64, size=0, dynamic_size = True)
        mb_dones = tf.TensorArray(tf.bool, size=0, dynamic_size = True)
        mb_mu = tf.TensorArray(tf.float64, size=0, dynamic_size = True)
        mb_L = tf.TensorArray(tf.float64, size=0, dynamic_size = True)

        #mb_obs, mb_rewards, mb_actions, mb_raw_actions, mb_values, mb_dones, mb_mu, mb_L = [],[],[],[],[],[],[],[]

        bufs = [mb_obs, mb_rewards, mb_actions, mb_raw_actions, mb_values, mb_dones, mb_mu, mb_L]
        n_step = 0
        
        def single_step(bufs, last_obs):
            mb_obs, mb_rewards, mb_actions, mb_raw_actions, mb_values, mb_dones, mb_mu, mb_L = bufs
            actions, raw_actions, values, mu, L = self.model.step(last_obs, training = self.training)

            # Append the experiences
            for n in range(len(mb_obs)):
                mb_obs[n] = mb_obs[n].write(n_step, last_obs[n])
            mb_actions = mb_actions.write(n_step, actions)
            mb_raw_actions = mb_raw_actions.write(n_step, raw_actions)
            mb_values = mb_values.write(n_step, values)

            #arrange penalties (not currently utilized)

            
            mb_mu = mb_mu.write(n_step, mu)
            mb_L = mb_L.write(n_step, L)

            # Take actions in env and record the results
            new_obs, rewards, dones = self.env.step(actions)

            mb_rewards = mb_rewards.write(n_step, rewards)
            mb_dones = mb_dones.write(n_step, dones)
            return [mb_obs, mb_rewards, mb_actions, mb_raw_actions, mb_values, mb_dones, mb_mu, mb_L], new_obs

        if until_done:
            done = tf.constant(False)
            while not done:
                bufs, last_obs = single_step(bufs, last_obs)
                done = tf.reduce_any(bufs[5].read(n_step))
                n_step += 1
        else:
            for n in range(self.nsteps):
                bufs, last_obs = single_step(bufs, last_obs)
                n_step += 1

        mb_obs, mb_rewards, mb_actions, mb_raw_actions, mb_values, mb_dones, mb_mu, mb_L = bufs
        # batch of lists of steps to list of batches
        # mb_obs = list(zip(*mb_obs))
        # list of batches of steps to list of batches of rollouts
        mb_obs = [sf01(mb_obs[i]) for i in range(len(self.obs_shape))]
        mb_actions = sf01(mb_actions)
        mb_raw_actions = sf01(mb_raw_actions)
        mb_values = tf.transpose(mb_values.stack(), (1,0,2))
        
        mb_mu = sf01(mb_mu)
        mb_L = sf01(mb_L)
        

        if self.gamma > 0.0 and self.training:
            # Discount/bootstrap off value fn
            last_values = self.model.value(last_obs)
            last_values = tf.squeeze(last_values,-1)
            done_in_end = mb_dones.read(n_step - 1) == [False]
            bstrap_rewards = mb_rewards.write(n_step, last_values)
            bstrap_rewards = bstrap_rewards.stack()
            mb_rewards = bstrap_rewards[:-1]
            app = tf.fill(mb_dones.element_shape, False)
            bstrap_dones = mb_dones.write(n_step, app)
            bstrap_dones = bstrap_dones.stack()
            mb_dones = bstrap_dones[:-1]
            rewards_done = get_discounted_rewards(mb_rewards, mb_dones, self.gamma)
            rewards_not_done = get_discounted_rewards(bstrap_rewards, bstrap_dones, self.gamma)[:-1]
            mb_rewards = tf.where(done_in_end, rewards_done, rewards_not_done)
        else:
            mb_rewards = mb_rewards.stack()
        
        mb_rewards = tf.transpose(mb_rewards, (1,0))
        mb_rewards = tf.reshape(mb_rewards, [-1])
        mb_values = tf.reshape(mb_values, [-1])
        bufs = [mb_obs, mb_rewards, mb_actions, mb_raw_actions, mb_values, mb_mu, mb_L]
        return bufs

    

# @tf.function(experimental_relax_shapes=True)
def sf01(tensorarray):
    """
    stack tensors, swap and then flatten axes 0 and 1
    """
    tensor = tensorarray.stack()
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
    return tf.reshape(tf.transpose(tensor, perm), (s[0] * s[1],) + tuple(tf.unstack(s[2:])))

def get_discounted_rewards(rewards, dones, gamma):
    gamma = tf.cast(gamma, tf.float64)
    rewards = tf.cast(rewards, tf.float64)
    dones = tf.cast(dones, tf.float64)
    one = tf.constant((1.,),dtype=tf.float64)
    initializer = tf.zeros(rewards.shape[1], dtype = tf.float64)
    discounted = tf.scan(
        lambda acc, rew_done: rew_done[0] + gamma * acc * (one - rew_done[1]),
        (rewards, dones),
        initializer,
        reverse = True)
    return discounted


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    