# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:12:15 2020

@author: Aapo Kössi
"""

import constants
import tensorflow as tf
import keras_tuner as kt
import tensorflow_probability as tfp
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.activations import swish
from tensorflow_probability.python.distributions import MultivariateNormalTriL as MVN



        

class deterministic_dropout(tf.keras.layers.Layer):
    def __init__(self, p_drop, **kwargs):
        super(deterministic_dropout, self).__init__(**kwargs)
        self.log_p = tf.math.log([[p_drop, 1 - p_drop]])
        
    def call(self, inputs):
        s = tf.shape(inputs)
        seed1 = tf.cast(tf.math.floormod(tf.math.reduce_sum(inputs) * 10002563.0 + 12875167.0, 349963), tf.int32)
        seed2 = tf.cast(tf.math.floormod(tf.math.reduce_max(inputs[:, -1]) * 15005 + 8371, 19993), tf.int32 )
        log_p = tf.tile(self.log_p, (s[0],1))
        mask = tf.random.stateless_categorical(log_p, s[1], [seed1, seed2]) == 1
        return tf.where(mask, inputs, tf.zeros_like(inputs))
        

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
        closes = ochlv[... , constants.others['data_index'].get_loc('prccd') ]
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
    
    """
    def __init__(self, output_shape, hp):
        """
        inputs: 1, output shape of actor network, essentially n_stocks that is desirable to be traded
                2, hyperparameters for the model, which impact the model architecture and size
        """
        super(Trader, self).__init__()
        
        #initializing the model layers
        self.arrange = Arranger()
        self.normalizer = Lambda(self.normalize_features)
        self.flatten_features_and_days = Lambda(lambda x: tf.reshape(x, [ x.shape[0] , x.shape[1] , -1 ]), trainable = False)
        self.encoder = self.make_mlp(hp)
        self.dense_temporal = self.make_temporal_DNN(hp)
        self.price_scaler = Lambda(lambda args: self.scale_prices(*args), trainable = False)
        self.equity_scaler = Lambda(lambda args: self.scale_equity(*args), trainable = False)
        self.common, common_out_shape = self.make_common_dense(output_shape, hp)
        self.cholesky_from_z = Cholesky_from_z(output_shape)
        self.clip_z = Lambda(self.clip_by_value_reverse_gradient)
        self.actor = self.make_actor(common_out_shape, output_shape, (self.cholesky_from_z, self.clip_z), hp)
        self.critic = Dense(1, name = 'critic')
        self.buy_limiter = Buy_limiter()
        self.convert_to_nshares = Lambda(self.cash_to_shares)
      

    def call(self, inputs, training=False, val_only = False, dist_features = False):

        #generating simplest bounds for action space
        #these are be used to scale the outputs of the policy network

        # inputs = [tf.cast(x, tf.float64) for x in inputs]

        [tf.debugging.check_numerics(x, f'input {n} not finite') for n, x in enumerate(inputs)]
        
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
        scaled_p = self.normalizer((scaled_p, 1))

        #dense network for recognising temporal features, convolutional might be more effective
        y = self.normalizer((y, 2))
        # y = self.flatten_features_and_days(y)
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
    def make_temporal_DNN(hp):
        architecture = hp.Choice('temporal_nn_type', ['Conv1D', 'LSTM'], default = 'Conv1D')
        model = tf.keras.Sequential(name = 'temporal_network')
        input_len = hp.Int('input_days', min_value= 10, max_value = 100, step = 5)
        if architecture == 'Conv1D':
            max_kernel_size = input_len - 6
            for n in range(hp.Int('conv_layers', min_value = 2, max_value = 6, default = 3, parent_name = 'temporal_nn_type', parent_values = ['Conv1D'])):
                filters = hp.Int(f'conv{n}_filters', min_value = 8, max_value = 256, sampling = 'log', default = 2**(n+5), parent_name = 'temporal_nn_type', parent_values = ['Conv1D'])
                kernel_size = hp.Int(f'conv{n}_kernel_size', min_value = 2, max_value = max_kernel_size, sampling = 'log', default = 5, parent_name = 'temporal_nn_type', parent_values = ['Conv1D'])
                padding = hp.Choice(f'conv{n}_padding', ['valid','same'], default = 'valid', parent_name = 'temporal_nn_type', parent_values = ['Conv1D'])
                model.add(Conv1D(filters, kernel_size, padding=padding, activation = swish))
                model.add(MaxPool2D(pool_size=(1,2),strides = (1,2)))
                if padding == 'same':
                    input_len = input_len
                    new_max_kernel_size = max_kernel_size // 2
                else:
                    input_len = (input_len - kernel_size + 1) // 2
                    new_max_kernel_size = input_len - 2
                if new_max_kernel_size < 2: break
                max_kernel_size = new_max_kernel_size
            model.add(tf.keras.layers.Flatten())
            for n in range(hp.Int('postconv_fc_layers', min_value = 0, max_value = 4, default = 2, parent_name = 'temporal_nn_type', parent_values = ['Conv1D'])):
                model.add(Dense(hp.Int(f'postconv{n}_units', min_value = 32, max_value = 1024, step = 32, default = 256 / 2**n, parent_name = 'temporal_nn_type', parent_values = ['Conv1D']), activation = swish))
                # model.add(deterministic_dropout(hp.Choice('p_drop1', [0.0, 0.1, 0.3, 0.6], default = 0.3, parent_name = 'temporal_nn_type', parent_values = ['Conv1D']), name = f'dropout{n}'))

        else:
            #transpose so that time dimension is first (after batch)
            #flatten channels and stocks
            model.add(Lambda(lambda x: tf.transpose(x, [0,2,1,3])))
            model.add(Lambda(lambda x: tf.reshape(x, tuple(tf.unstack(tf.shape(x)[:-2])) + (-1,))))
            for n in range(hp.Int('lstm_layers', min_value = 1, max_value = 4, default = 2, parent_name = 'temporal_nn_type', parent_values = ['LSTM']) - 1):
                units = hp.Int(f'lstm{n}_units', min_value = 256 // 4, max_value = 2048 // 4, step = 256 // 4, default = 512 // 4, parent_name = 'temporal_nn_type', parent_values = ['LSTM'])
                model.add(LSTM(units, return_sequences = True, name = f'temporal_{n}'))
            final_units = hp.Int('last_lstm_units', min_value = 256 // 4, max_value = 2048 // 4, step = 256 // 4, default = 512 // 4, parent_name = 'temporal_nn_type', parent_values = ['LSTM'])
            model.add(LSTM(final_units, name = 'last_temporal'))

        # model.add(Conv1D(32, 5, activation = swish))
        # model.add(MaxPool2D(pool_size = (1,2), strides = (1,2)))
        # model.add(Conv1D(64, 5, activation = swish))
        # model.add(MaxPool2D(pool_size = (1,2), strides = (1,2)))
        # model.add(Conv1D(128, 5, activation = swish))
        # model.add(MaxPool2D(pool_size = (1,2), strides = (1,2)))
        # model.add(tf.keras.layers.Flatten())
        # model.add(Dense(128, activation = swish))
        # model.add(deterministic_dropout(hp.Choice('p_drop1', [0.0, 0.1, 0.3, 0.6], default = 0.3)))
        # model.add(Dense(100, activation = swish))
        # model.add(deterministic_dropout(hp.Choice('p_drop2', [0.0, 0.1, 0.3, 0.6], default = 0.3)))
        # model.add(Dense(64, activation = swish))
        return model
    
    @staticmethod
    def make_mlp(hp):
        """
        Returns
        -------
        model : TF MODEL WITH OUTPUT SHAPE (BATCH, 2*prod(dim[1:]))???.
    
        """
        model = tf.keras.Sequential(name = 'autoencoder')
        model.add(tf.keras.layers.Flatten())
        for n in range(hp.Int('mlp_layers', min_value = 1, max_value = 3, default = 2)):
            units = hp.Int(f'mlp{n}_units', min_value = 10, max_value = 100, step = 10, default = 32 // 2**n)
            model.add(Dense(units, activation = swish, name = f'autoencoder_{n}'))

        return model
    
    @staticmethod
    def scale_prices(lasts, capital):
        return lasts / (capital + 1e-8)

    @staticmethod
    def scale_equity(equity, scaled_prices):
        return scaled_prices * equity
    

    def make_common_dense(self, output_shape, hp):
        model = tf.keras.Sequential(name = 'common')
        for n in range(hp.Int('common_layers', min_value = 1, max_value = 4, default = 3)):
            units = hp.Int(f'common{n}_units', min_value = 1, max_value = 129, step = 4, default = 32 // 2**n)
            model.add(Dense(output_shape * units, activation = swish, name = f'common{n}'))
        return model, (output_shape * units,)
    
    @staticmethod
    def make_actor(input_shape, num_outputs, shapers, hp):
        def scale(unscaled, lb, ub):
            return unscaled * (ub - lb) - lb
        
            
        gen_cholesky, clip_z = shapers

        input_main = tf.keras.Input(shape = input_shape)
        lb, ub = (tf.keras.Input(shape = (num_outputs,)), tf.keras.Input(shape = (1,)))
        inputs = (input_main, lb, ub)
        hidden_input = input_main
        n_hidden_layers = hp.Int('n_actor_hidden_layers', min_value = 1, max_value = 4, default = 1)
        for n_layer in range(n_hidden_layers):
            latest_hidden = Dense(num_outputs * hp.Int(f'actor_hidden{n_layer}_units', min_value = 2, max_value = 16, step = 2, default = 8),
                          activation = swish, name = f'actor_common{n_layer}')(hidden_input)
            hidden_input = latest_hidden
        mu_hidden = Dense(num_outputs * hp.Int('mu_hidden_units', min_value = 2, max_value = 16, step = 2, default = 4),
                          activation = swish, name = 'mu_hidden')(latest_hidden)

        
        #get means bound near the observation space
        buy_mu = Dense(num_outputs, activation = swish, name = 'mu_buys')(mu_hidden)
        sell_mu = Dense(num_outputs, activation = swish, name = 'mu_sells')(mu_hidden)
        unscaled_mu = tf.where(buy_mu  > sell_mu, buy_mu, - sell_mu) #choose bias based on absolute value of buy vs sell
        # unscaled_mu = buy_mu - sell_mu
        # unscaled_mu = unscaled_mu + constants.MIN_SWISH #min(swish(x)) = -0.278464..., we need output >= 0
        mu = unscaled_mu * (ub - lb)
        
        #generate lower triangle scale matrix
        z_vec = Dense(num_outputs*(num_outputs - 1) / 2, activation = 'tanh', name = 'z_layer')(latest_hidden)
        z_vec_clipped = clip_z(z_vec)
        std_vec = Dense(num_outputs, activation = swish, name = 'stdev_layer')(latest_hidden)
        # std_vec = std_vec + constants.MIN_SWISH # diagonal part has to be positive
        #clip std between 0 and range of possible actions * l_scale_hparam
        l_scale = hp.Float('std_scale', min_value = 0.1, max_value = 5., sampling = 'log', default = constants.l_scale)
        std_vec = tfp.math.clip_by_value_preserve_gradient(std_vec, 0.0, 1.0) * (ub - lb) * l_scale
        
        # shape the vectors into a lower triangular matrix
        L = gen_cholesky(z_vec_clipped)
        W = tf.linalg.diag(std_vec) #rescale from L_correlation to L_covariance, units capital
        L = tf.matmul(W, L)
        outputs = (mu, L)
        
        model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'actor_fc_network')
        return model
    
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
    
class HyperTrader(kt.HyperModel):
    def __init__(self, out_shape):
        super(HyperTrader, self).__init__()
        self.out_shape = out_shape
    
    def build(self, hp):
        model = Trader(self.out_shape, hp)
        return model

        
    
    
    
    
    
    