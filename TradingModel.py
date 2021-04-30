# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:12:15 2020

@author: Aapo Kössi
"""

import constants
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import swish
from tensorflow_probability.python.distributions import MultivariateNormalTriL as MVN


#TODO: implement hparam optimization
#TODO: something happens between first and second call that fucks up raw_actions or mu.
# HOW THE F UCK???????????++++??????????+++?++++??+

        
class Cholesky_from_z(tf.keras.layers.Layer):
    def __init__(self, output_shape):
        super(Cholesky_from_z, self).__init__(trainable = False)
        self.size = output_shape
        # generate indices corresponding to the lower indices
        l_indices = []
        for i in range(self.size):
            for j in range(self.size):
                if j < i: l_indices.append([i,j])
        self.l_indices = tf.constant(l_indices)
        
        
    #can either map using TensorArrays or tf.scan, very similar performance on CPU
    def call(self, inputs):
        # tf.debugging.assert_all_finite(inputs, 'chol input not finite')
        L = tf.map_fn(lambda x: self.scan_accumulate_L(self.shaper(x)),inputs)
        # tf.print(tf.reduce_min(tf.linalg.diag_part(L)))
        return L

    def shaper(self, vector):
        return tf.scatter_nd(self.l_indices, vector, (self.size, self.size))
        
    def accumulate_L(self, z):
        j_ta = tf.TensorArray(tf.float64, size =0, dynamic_size=True, clear_after_read=True, element_shape= [self.size])
        for j in range(self.size):
            z_j = z[:,j]
            uppers = tf.zeros(j, dtype = tf.float64)
            diag = tf.sqrt(1. - tf.math.reduce_sum(j_ta.stack() ** 2, axis = 0)[j])
            diag = tf.expand_dims(diag, 0)
            lower = z_j[j+1:] * tf.sqrt(1. - tf.math.reduce_sum(j_ta.stack() ** 2, axis = 0)[j+1:])
            j_ta = j_ta.write(j,tf.concat((uppers, diag, lower), 0))
        L = tf.transpose(j_ta.stack())
        return L

    def scan_accumulate_L(self, z):
        # tf.debugging.assert_all_finite(z, 'got non finite z')
        initializer = (tf.zeros([self.size],dtype = tf.float64), tf.zeros([self.size],dtype = tf.float64))
        def scan_func(a, t):
            uppers = tf.zeros(t[1], dtype = tf.float64)
            diag = tf.sqrt(1. - a[1])[t[1]]
            diag = tf.expand_dims(diag, 0)
            lower = t[0][t[1]+1:] * tf.sqrt(1. - a[1])[t[1]+1:]
            col = tf.concat([uppers, diag, lower],0)
            col = tf.ensure_shape(col, [self.size])
            # col = tf.debugging.assert_all_finite(col, 'column of L not finite')
            return col, a[1] + col ** 2
        Lt = tf.scan(scan_func, (tf.transpose(z), tf.range(self.size,dtype=tf.int32)), initializer=initializer, parallel_iterations=16, infer_shape=False)
        L = tf.transpose(Lt[0])
        L = tf.ensure_shape(L, [self.size,self.size])
        tf.debugging.assert_near(tf.linalg.diag_part(tf.matmul(L,L,transpose_b=True)),tf.ones(self.size,dtype=tf.float64), message = f'{tf.linalg.diag_part(L)}')
        return L


        
    
class Buy_limiter(tf.keras.layers.Layer):
    def __init__(self):
        super(Buy_limiter, self).__init__(trainable = False)
        
    def call(self, inputs):
        """
        0: the raw actions
        1: the available cash
        2: equity prices
        
        out: buying actions scaled so that (most) possible cash is expended
        """
        pos_mask = inputs[0] > 0.0
        needed_capital = tf.reduce_sum(tf.where(pos_mask, inputs[0] * inputs[2],tf.zeros_like(inputs[0])),axis=1,keepdims=True)
        ratios = tf.math.divide(needed_capital, inputs[1] + 1e-8)
        on_margin = ratios >= 0.9
        scaled = tf.math.divide(inputs[0], ratios * (1 + 1e-1) + 1e-3) #scaled could remain too large in extreme cases without eps here
        cond = tf.math.logical_and(pos_mask, on_margin)
        out = tf.where(cond, scaled, inputs[0])
        return out, on_margin
            

class Arranger(tf.keras.layers.Layer):
    def __init__(self):
        super(Arranger, self).__init__(trainable = False)
        self.nil = tf.constant(0, dtype = tf.float64)

    def call(self, inputs):
        """
        
        Parameters
        ----------
        inputs : stock data in shape (n_batch, n_ticker,) + (x,)

        arranges  symbols intra-batch, according to the sorted performances
        """
        ochlv = inputs[2]
        shape = ochlv.shape
        closes = ochlv[... , constants.others['data_index'].get_loc('close') ]
        condition = tf.not_equal(closes, self.nil)
        ragged = tf.ragged.boolean_mask(closes, condition)
        starts = ragged.to_tensor(default_value = 0, shape = (shape[:-1]))[... ,0]
        last_values = closes[... ,-1]
        performances = tf.math.divide_no_nan(last_values - starts, starts)
        
        orders = tf.argsort(performances, direction = 'DESCENDING')
        out =  tuple([tf.gather(elem, orders, batch_dims = 1) for elem in inputs])
        return out, orders
    

class Trader(tf.keras.Model):
        
    """
    inputs: 1st encoded categories, 2nd equity, 3rd stock data, 4th last prices, 5th current capital
    
    
    TODO: HPARAM TUNING AND GRAPH EXECUTION
    
    """
    def __init__(self, output_shape):
        super(Trader, self).__init__()
        
        action_dim = output_shape[-1]
        
        #initializing the model layers
        self.arrange = Arranger()
        self.normalizer = tf.keras.layers.Lambda(self.normalize_features)
        self.flatten_features_and_days = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [ x.shape[0] , x.shape[1] , -1 ]), trainable = False)
        self.encoder = self.make_mlp()
        self.dense_temporal = self.make_temporal_DNN()
        self.price_scaler = tf.keras.layers.Lambda(lambda args: self.scale_prices(*args), trainable = False)
        self.equity_scaler = tf.keras.layers.Lambda(lambda args: self.scale_equity(*args), trainable = False)
        self.common = self.make_common_dense(action_dim)
        self.cholesky_from_z = Cholesky_from_z(action_dim)
        self.clip_z = tf.keras.layers.Lambda(self.clip_by_value_reverse_gradient)
        self.actor = self.make_actor(action_dim, (self.cholesky_from_z, self.clip_z))
        self.critic = Dense(1, name = 'critic')
        self.buy_limiter = Buy_limiter()
        self.convert_to_nshares = tf.keras.layers.Lambda(self.cash_to_shares)
      

    def call(self, inputs, training=False, val_only = False, dist_features = False):

        #generating simplest bounds for action space
        #these are be used to scale the outputs of the policy network

        inputs = [tf.cast(x, tf.float64) for x in inputs]
        [tf.debugging.check_numerics(x, 'input not finite') for x in inputs]
        
        (x, e, y, p), orders = self.arrange(inputs[:4])
        c = inputs[4]
        lb = - tf.stack(p * e)
        ub = c
        # tf.print(e, summarize = 160)
        # tf.print(tf.reduce_min(p))
        # tf.print(tf.reduce_max(lb))
        # tf.print(tf.reduce_min(ub))
        tf.debugging.assert_non_negative(e,'equity negative')
        tf.debugging.assert_non_negative(c, 'capital negative')
        #simple dense network for feature encoding of categorical data
        x = self.encoder(x)
        # tf.debugging.assert_all_finite(x, 'autoencoder output not finite')
        #rescaling current equity and stock prices for the main network
        scaled_p = self.price_scaler((p, c))
        scaled_e = self.equity_scaler([e, scaled_p])

        #dense network for recognising temporal features, convolutional might be more effective
        y = self.normalizer((y, 2))
        y = self.flatten_features_and_days(y)
        y = self.dense_temporal(y)
        # tf.debugging.assert_all_finite(y, 'temporal net output not finite')
        #concatenates the different inputs and categorizes features across all stocks
        tf.debugging.assert_all_finite(scaled_e, 'scaled equity not finite')
        tf.debugging.assert_all_finite(scaled_p, 'scaled prices not finite')
        concat = tf.keras.layers.concatenate((x, scaled_e, y, scaled_p))
        # tf.debugging.assert_all_finite(concat, 'concat output not finite')
        main = self.common(concat)
        # tf.debugging.assert_all_finite(main, 'main network not finite')
        #value network
        value = self.critic(main)
        if val_only:
            return value
        
        #actor network
        mu, L = self.actor((main, lb, ub))
        # tf.print(tf.reduce_min(tf.linalg.diag_part(L)))
        # tf.debugging.assert_all_finite(mu, 'action means not finite??')
        mu = self.convert_to_nshares((mu, p))
        L = tf.matmul(tf.math.divide_no_nan(tf.constant(1.0, dtype = tf.float64), tf.linalg.diag(p)), L)
        L_epsilon = tf.linalg.eye(mu.shape[-1], dtype = tf.float64) * constants.l_epsilon
        L = L + L_epsilon


        tf.debugging.assert_positive(c, message='negative capital')
        tf.debugging.assert_non_negative(e,message='negative equity')

        # tf.debugging.assert_all_finite(L, 'got non finite L')
        
        #limiting the mean actions to be in bounds
        mu, neg_cash = self.buy_limiter((mu, c, p))
        final_mu = self.clip_selling((mu, -e))
        # tf.print(tf.reduce_min(e + final_mu))
        
        
        if dist_features:
            return final_mu, L, value

        #creating the gaussian and sampling an action
        dist = MVN(loc = final_mu, scale_tril = L)
        raw_action = dist.sample()
        
        # dist2 = MVN(loc = final_mu, scale_tril = L)
        # tf.print(tf.cast(dist2.log_prob(raw_action),tf.int32))
        #ensuring that sample is also in bounds
        action, _ = self.buy_limiter((raw_action, c, p))
        action = self.clip_selling((action, -e))  

        # dummy penalties for trying to short, not possible with current model architecture
        neg_shares = tf.fill(c.shape, False) 

        #reordering the actions back into the input order
        action = tf.gather(action, tf.argsort(orders), batch_dims = 1)
        
        if training:
            return action, raw_action, value, neg_cash, neg_shares, final_mu, L

        action_means = tf.gather(final_mu, tf.argsort(orders), batch_dims = 1)
        return action_means

    def value(self, obs):
        vpreds = self.call(obs, val_only = True)
        return vpreds
    
    
    @staticmethod
    def make_temporal_DNN():
        model = tf.keras.Sequential(name = 'temporal_network')
        model.add(Dense(64, activation = swish))
        model.add(Dense(32, activation = swish))
        model.add(Dense(16, activation = swish))
        model.add(tf.keras.layers.Flatten())
        return model
    
    @staticmethod
    def make_mlp():
        """
        Returns
        -------
        model : TF MODEL WITH OUTPUT SHAPE (BATCH, 2*prod(dim[1:]))???.
    
        """
        model = tf.keras.Sequential(name = 'autoencoder')
        model.add(tf.keras.layers.Flatten())
        model.add(Dense(32, activation = swish, name = 'autoencoder_1' ))
        model.add(Dense(16, activation = swish, name = 'autoencoder_2' ))
        return model
    
    @staticmethod
    def scale_prices(lasts, capital):
        return lasts / (capital + 1e-8)

    @staticmethod
    def scale_equity(equity, scaled_prices):
        return scaled_prices * equity
    

    def make_common_dense(self, output_shape):
        model = tf.keras.Sequential(name = 'common')
        model.add(Dense(output_shape * 32, activation = swish, name = 'common1'))
        model.add(Dense(output_shape * 16, activation = swish, name = 'common2'))
        model.add(Dense(output_shape * 8, activation = swish, name = 'common3'))
        return model
    
    @staticmethod
    def make_actor(num_outputs, shapers):

        def scale(unscaled, lb, ub):
            return unscaled * (ub - lb) - lb
            
        gen_cholesky, clip_z = shapers

        input_main = tf.keras.Input(shape = (num_outputs * 8,))
        lb, ub = (tf.keras.Input(shape = (num_outputs,)), tf.keras.Input(shape = (1,)))
        inputs = (input_main, lb, ub)
        hidden = Dense(num_outputs * 8, activation = swish, name = 'actor_common1')(input_main)
        hidden2 = Dense(num_outputs * 4, activation = swish, name = 'actor_hidden_mean')(hidden)

        
        #get means bound near the observation space
        buy_mu = Dense(num_outputs, activation = swish, name = 'mu_buys')(hidden2)
        sell_mu = Dense(num_outputs, activation = swish, name = 'mu_sells')(hidden2)
        unscaled_mu = tf.where(buy_mu  > sell_mu, buy_mu, - sell_mu) #choose bias based on absolute value of mean buy vs sell
        # unscaled_mu = buy_mu - sell_mu
        # unscaled_mu = unscaled_mu + constants.MIN_SWISH #min(swish(x)) = -0.278464..., we need output >= 0
        mu = unscaled_mu * (ub - lb)
        
        #generate lower triangle scale matrix
        z_vec = Dense(num_outputs*(num_outputs - 1) / 2, activation = 'tanh', name = 'z_layer')(hidden)
        z_vec_clipped = clip_z(z_vec)
        std_vec = Dense(num_outputs, activation = swish, name = 'stdev_layer')(hidden)
        # std_vec = std_vec + constants.MIN_SWISH # diagonal part has to be positive
        #clip std between 0 and range of possible actions * l_scale_hparam
        std_vec = tfp.math.clip_by_value_preserve_gradient(std_vec, 0.0, 1.0) * (ub - lb) * constants.l_scale
        
        # shape the vectors into a lower triangular matrix
        L = gen_cholesky(z_vec_clipped)
        W = tf.linalg.diag(std_vec) #rescale from L_correlation to L_covariance, units capital
        L = tf.matmul(W, L)
        outputs = (mu, L)
        
        model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'actor_fc_network')
        return model
    
    
    def selling_more_than_available(self, raw_actions, lb):
        """
        DEPRECATED
        
        """
        flag = tf.fill(raw_actions.shape[:-1], False)
        for i in range(raw_actions.shape[0]):
            for j in range(raw_actions.shape[1]):  
                if raw_actions[i,j] < lb[i,j] and raw_actions[i,j] < 0:
                    flag = tf.tensor_scatter_nd_update(flag, [[i]], [True])
        return flag
    
    @staticmethod
    def cash_to_shares(args):
        return tf.math.divide_no_nan(args[0], args[1])

    
    @staticmethod
    @tf.custom_gradient
    def clip_by_value_reverse_gradient(inputs):
        # tf.debugging.assert_all_finite(inputs, 'unclipped z not finite??')
        ub = tf.cast(constants.TANH_UB, tf.float64)
        out = tf.clip_by_value(inputs, -ub, ub)
        # tf.debugging.assert_all_finite(out, 'clipped z not finite??')
        def grad(dy):
            return tf.where(tf.math.logical_or(inputs <= -ub, inputs >= ub), -dy, dy)
        return out, grad
    
    @staticmethod
    @tf.custom_gradient
    def clip_selling(inputs):
        val = inputs[0]
        min_val = inputs[1]
        def grad(upstream):
            d_clipped_d_val = tf.ones_like(val)
            d_clipped_d_min = tf.zeros_like(min_val)
            return upstream * d_clipped_d_val, upstream * d_clipped_d_min
        return tf.math.maximum(val, min_val), grad
    
    @staticmethod
    def normalize_features(args):
        inputs, axis = args
        means = tf.reduce_mean(inputs, axis, keepdims=True)
        stdevs = tf.math.reduce_std(inputs, axis, keepdims=True)
        return tf.math.divide_no_nan((inputs - means), stdevs)
        
    
    
    
    
    
    