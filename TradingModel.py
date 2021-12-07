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
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.activations import swish
from tensorflow_probability.python.distributions import MultivariateNormalTriL as MVN


class deterministic_dropout(tf.keras.layers.Layer):
    def __init__(self, p_drop, hp, **kwargs):
        super(deterministic_dropout, self).__init__(**kwargs)
        self.p = p_drop
        log_p = tf.math.log([p_drop, 1 - p_drop])
        self.log_p_drop = tf.expand_dims(log_p, 0)

    def call(self, inputs):
        num_samples = tf.shape(inputs)[1]

        def map_fn(elems):
            inputs, seed = elems
            dropmask = tf.cast(tf.random.stateless_categorical(self.log_p_drop, num_samples, seed), tf.bool)
            out = tf.where(dropmask, tf.zeros_like(inputs), inputs)
            return out

        seed1 = tf.cast(tf.math.floormod(tf.math.reduce_sum(inputs, axis=-1) * 10002563.0 + 12875167.0, 349963.0),
                        tf.int32)
        seed2 = tf.cast(tf.math.floormod(tf.math.reduce_max(inputs, axis=-1) * 15005.0 + 8371.0, 11351.0), tf.int32)
        seed = tf.transpose(tf.squeeze([seed1, seed2]))
        out = tf.vectorized_map(map_fn, (inputs, seed))
        out = tf.squeeze(out)
        return out


class Cholesky_from_z(tf.keras.layers.Layer):
    def __init__(self, output_shape):
        super(Cholesky_from_z, self).__init__(trainable=False)
        self.size = output_shape
        # generate indices corresponding to the lower indices
        l_indices = []
        for i in range(self.size):
            for j in range(self.size):
                if j < i: l_indices.append([i, j])
        self.l_indices = tf.expand_dims(tf.constant(l_indices), 0)

    # can either map using TensorArrays or tf.scan, very similar performance on CPU
    def call(self, inputs):
        # tf.debugging.assert_all_finite(inputs, 'chol input not finite')
        L = self.accumulate_squared_L(self.shaper(inputs))
        # L = tf.map_fn(lambda x: self.accumulate_L(self.shaper(x)),inputs)
        # tf.print(tf.reduce_min(tf.linalg.diag_part(L)))
        return L

    def shaper(self, vector):
        batch_size = tf.shape(vector)[0]
        batch_repeats = self.l_indices.shape[1]
        batch_index = tf.repeat(tf.expand_dims(tf.expand_dims(tf.range(0, batch_size), -1), -1), batch_repeats, 1)
        batched_l_indices = tf.concat((batch_index, tf.repeat(self.l_indices, batch_size, 0)), -1)
        return tf.scatter_nd(batched_l_indices, vector, (batch_size, self.size, self.size))

    def accumulate_squared_L(self, z):
        signs = tf.math.sign(z)
        signs = tf.where(signs == 0, tf.cast(1., tf.float32), signs)
        z_squared = z * z
        x_squared = tf.zeros_like(z)

        def iterate(j, x_squared):
            z_col = z_squared[..., j]
            uppers = tf.zeros(tf.shape(z)[:-1], dtype=tf.float32)[..., :j]
            diag = [1.] - tf.math.reduce_sum(x_squared[..., j], axis=-1, keepdims=True)
            lower = z_col[..., j + 1:] * (1. - tf.math.reduce_sum(x_squared[..., j + 1:, :], axis=-1))
            col = tf.transpose(tf.expand_dims(tf.concat((uppers, diag, lower), -1), -2), perm=[0, 2, 1])
            mat = tf.pad(col, [[0, 0], [0, 0], [j, self.size - 1 - j]])
            x_squared += mat
            return x_squared

        for j in range(self.size):
            x_squared = iterate(j, x_squared)
        return tf.sqrt(x_squared) * signs


class Buy_limiter(tf.keras.layers.Layer):
    def __init__(self):
        super(Buy_limiter, self).__init__(trainable=False)

    def call(self, inputs):
        """
        0: the raw actions
        1: the available cash
        2: equity prices
        
        out: buying actions scaled so that (most) possible cash is expended
        """
        pos_mask = inputs[0] > 0.0
        needed_capital = tf.reduce_sum(tf.where(pos_mask, inputs[0] * inputs[2], tf.zeros_like(inputs[0])), axis=1,
                                       keepdims=True)
        ratios = tf.math.divide(needed_capital, inputs[1] + 1e-8)
        on_margin = ratios >= 0.9
        scaled = tf.math.divide(inputs[0], ratios * (
                    1 + 1e-1) + 1e-3)  # scaled could remain too large in extreme cases without eps here
        cond = tf.math.logical_and(pos_mask, on_margin)
        out = tf.where(cond, scaled, inputs[0])
        return out


class Arranger(tf.keras.layers.Layer):
    def __init__(self, close_idx):
        super(Arranger, self).__init__(trainable=False)
        self.nil = tf.constant(0, dtype=tf.float32)
        self.close_idx = tf.constant(close_idx)

    def call(self, inputs):
        """
        
        Parameters
        ----------
        inputs : stock data in shape (n_batch, n_ticker,) + (x,)

        arranges  symbols intra-batch, according to the sorted performances
        """
        ochlv = inputs[2]
        shape = ochlv.shape
        closes = ochlv[..., self.close_idx]
        starts = closes[..., 0]
        last_values = closes[..., -1]
        performances = tf.math.divide_no_nan(last_values - starts, starts)

        orders = tf.argsort(performances, direction='DESCENDING')
        out = tuple([tf.gather(elem, orders, batch_dims=1) for elem in inputs])
        return out, orders


class Trader(tf.keras.Model):
    """
    inputs to call: 1st encoded categories, 2nd equity, 3rd stock data, 4th last prices, 5th current capital
    
    """

    def __init__(self, output_shape, hp, close_idx):
        """
        PARAMETERS: 1, output shape of actor network, essentially n_stocks that is desirable to be traded
                2, hyperparameters for the model, which impact the model architecture and size
                3, channel number for the daily close price in ohlcvd data
        RETURNS: tf.keras.Model object
        """
        super(Trader, self).__init__()

        # initializing the model layers
        self.arrange = Arranger(close_idx)
        self.normalizer = Lambda(self.normalize_features)
        self.flatten_features_and_days = Lambda(lambda x: tf.reshape(x, [x.shape[0], x.shape[1], -1]), trainable=False)
        self.encoder = self.make_mlp(hp)
        self.temporal = self.make_temporal_DNN(output_shape, hp)
        self.price_scaler = Lambda(lambda args: self.scale_prices(*args), trainable=False)
        self.equity_scaler = Lambda(lambda args: self.scale_equity(*args), trainable=False)
        self.common, common_out_shape = self.make_common_dense(output_shape, hp)
        self.cholesky_from_z = Cholesky_from_z(output_shape)
        self.clip_z = Lambda(self.clip_by_value_reverse_gradient)
        self.actor = self.make_actor(common_out_shape, output_shape, (self.cholesky_from_z, self.clip_z), hp)
        self.critic = Dense(1, name='critic')
        self.buy_limiter = Buy_limiter()
        self.convert_to_nshares = Lambda(self.cash_to_shares)

    def call(self, inputs, training=False, val_only=False, dist_features=False):

        # generating simplest bounds for action space
        # these are be used to scale the outputs of the policy network
        (x, e, y, p), orders = self.arrange(inputs[:4])
        c = inputs[4]
        lb = - tf.stack(p * e)
        ub = c

        # simple dense network for feature encoding of categorical data
        x = self.encoder(x)

        # rescaling current equity and stock prices for the main network
        scaled_p = self.price_scaler((p, c))
        scaled_e = self.equity_scaler([e, scaled_p])
        scaled_p = self.normalizer((scaled_p, 1))

        # dense network for recognising temporal features, convolutional might be more effective
        y = self.normalizer((y, 2))
        y = self.temporal(y)

        # concatenates the different inputs and categorizes features across all stocks
        concat = tf.keras.layers.concatenate((x, scaled_e, y, scaled_p))
        main = self.common(concat)
        # value network
        value = self.critic(main)
        if val_only:
            return value

        # actor network
        mu, L = self.actor((main, lb, ub))
        mu = self.convert_to_nshares((mu, p))

        L = tf.matmul(tf.math.divide_no_nan(tf.constant(1.0, dtype=tf.float32), tf.linalg.diag(p)), L)
        L_epsilon = tf.linalg.eye(mu.shape[-1], dtype=tf.float32) * constants.l_epsilon
        L = L + L_epsilon

        # limiting the mean actions to be in bounds
        mu = self.buy_limiter((mu, c, p))
        final_mu = self.clip_selling((mu, -e))

        if dist_features:
            return final_mu, L, value

        # creating the gaussian and sampling an action
        dist = MVN(loc=final_mu, scale_tril=L)
        raw_action = dist.sample()

        # ensuring that sample is also in bounds
        action = self.buy_limiter((raw_action, c, p))
        action = self.clip_selling((action, -e))

        # reordering the actions back into the input order
        action = tf.gather(action, tf.argsort(orders), batch_dims=1)

        if training:
            return action, raw_action, value, final_mu, L

        action_means = tf.gather(final_mu, tf.argsort(orders), batch_dims=1)
        return action_means

    def make_critic(self, in_shape):
        # unused
        inputs = tf.keras.Input(shape=in_shape)
        vpred = Dense(1)(inputs)
        stddev = Dense(1, activation='softplus')(inputs)
        out = tf.keras.layers.concatenate((vpred, stddev))
        return out

    def value(self, obs):
        vpreds = self.call(obs, val_only=True)
        return vpreds

    @staticmethod
    def make_temporal_DNN(output_shape, hp):

        architecture = hp.Choice('temporal_nn_type', ['Conv1D', 'LSTM'], default='Conv1D')
        bidirectional = hp.Choice('Bidirectional LSTM', [True, False], default=True)
        separated_post_conv = hp.Choice('separated_post_conv', [True, False], default = True)
        model = tf.keras.Sequential(name='temporal_network')
        input_len = hp.Int('input_days', min_value=60, max_value=120, step=10,
                           default=120)  # correct defaule 80

        conv_filters = []
        conv_kernel_sizes = []
        repeats = []
        paddings = []
        input_len_tracker = input_len
        max_kernel_size = 12
        max_conv_layers = 5
        conv_layers = False
        for n in range(max_conv_layers):
            conv_filters.append(hp.Int(f'conv{n}_filters', min_value=16 + 8*n, max_value=32+8*n**2,
                                       step=16, default=32+8*n**2)) # 2 ** (n + 6) // (n + 2)
            conv_kernel_sizes.append(
                hp.Int(f'conv{n}_kernel_size', min_value=2, max_value=max_kernel_size, default=max_kernel_size))
            repeats.append(hp.Choice(f'conv{n}_repeats', values=[0,1], default=1)) #1*(1<n<5)
            paddings.append(hp.Choice(f'conv{n}_padding', ['valid', 'same'], default='same'))
            if paddings[n] == 'same':
                input_len_tracker = input_len_tracker // 2
            else:
                input_len_tracker = (input_len_tracker - conv_kernel_sizes[n] + 1) // 2
            if max_kernel_size >= input_len_tracker and not conv_layers:
                conv_layers = n + 1
            max_kernel_size -= 3 - (n + 1) // 2

        separated_dense_units = hp.Int('separate_post_conv_units', min_value = 8*output_shape,
                                       max_value = 16*output_shape, step = 8, default = 16*output_shape)

        postconv_units = []
        max_postconv_layers = 3
        for n in range(max_postconv_layers):
            postconv_units.append(
                hp.Int(f'postconv{n}_units', min_value=32, max_value=192, step=32, default=192))  # 256 // 2**n
        postconv_layers = hp.Int(
            'postconv_fc_layers', min_value=0, max_value=max_postconv_layers, default=max_postconv_layers)  # 2

        max_lstm_layers = 4
        lstm_layers = hp.Int('lstm_layers', min_value=1, max_value=max_lstm_layers, default=max_lstm_layers)  # 3
        lstm_units = []
        for n in range(max_lstm_layers):
            lstm_units.append(hp.Int(f'lstm{n}_units', min_value=32, max_value=96, step=8,
                                     default=96))
        if bidirectional:
            for n in range(max_lstm_layers - 1):
                lstm_units[n] = lstm_units[n] // 2

        if architecture == 'Conv1D':
            for n in range(conv_layers):
                for _ in range(repeats[n]):
                    model.add(Conv1D(conv_filters[n], conv_kernel_sizes[n], padding='same', activation=swish))
                model.add(Conv1D(conv_filters[n], conv_kernel_sizes[n], padding=paddings[n], activation=swish))
                model.add(MaxPool2D(pool_size=(1, 2), strides=(1, 2)))

            if separated_post_conv:
                # flatten length and channels
                model.add(Lambda(lambda x: tf.reshape(x, tuple(tf.unstack(tf.shape(x)[:-2])) + (-1,))))
                # dense layer applied separately for each stock
                model.add(Dense(separated_dense_units, activation = swish))

            model.add(tf.keras.layers.Flatten())
            for n in range(postconv_layers):
                model.add(Dense(postconv_units[n], activation=swish))

        else:
            # transpose so that time dimension is first (after batch)
            # flatten channels and stocks
            model.add(Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3])))
            model.add(Lambda(lambda x: tf.reshape(x, tuple(tf.unstack(tf.shape(x)[:-2])) + (-1,))))
            for n in range(lstm_layers - 1):
                cell = LSTM(lstm_units[n], return_sequences=True, name=f'temporal_{n}')
                if bidirectional:
                    model.add(Bidirectional(cell))
                else:
                    model.add(cell)
            final_units = lstm_units[-1]
            model.add(LSTM(final_units, name='last_temporal'))

        return model

    @staticmethod
    def make_mlp(hp):
        """
        Returns
        -------
        model : TF MODEL WITH OUTPUT SHAPE (BATCH, units).
    
        """
        model = tf.keras.Sequential(name='autoencoder')
        model.add(tf.keras.layers.Flatten())
        for n in range(hp.Int('mlp_layers', min_value=1, max_value=3, default=3)): #2
            units = hp.Int(f'mlp{n}_units', min_value=8, max_value=64, step=8, default=64) # 32 // 2 ** n
            model.add(Dense(units, activation=swish, name=f'autoencoder_{n}'))

        return model

    @staticmethod
    def scale_prices(lasts, capital):
        return lasts / (capital + 1e-8)

    @staticmethod
    def scale_equity(equity, scaled_prices):
        return scaled_prices * equity

    def make_common_dense(self, output_shape, hp):
        model = tf.keras.Sequential(name='common')
        for n in range(hp.Int('common_layers', min_value=1, max_value=3, default=3)): # 2
            units = hp.Int(f'common{n}_units', min_value=16, max_value=128, step=16, default=128) # 128 // 2 ** n
            model.add(Dense(output_shape * units, activation=swish, name=f'common{n}'))
        return model, (output_shape * units,)

    @staticmethod
    def make_actor(input_shape, num_outputs, shapers, hp):

        gen_cholesky, clip_z = shapers

        input_main = tf.keras.Input(shape=input_shape)
        lb, ub = (tf.keras.Input(shape=(num_outputs,)), tf.keras.Input(shape=(1,)))
        inputs = (input_main, lb, ub)
        hidden_input = input_main
        n_hidden_layers = hp.Int('n_actor_hidden_layers', min_value=1, max_value=4, default=4) # 1
        for n_layer in range(n_hidden_layers):
            latest_hidden = Dense(
                num_outputs * hp.Int(f'actor_hidden{n_layer}_units', min_value=2, max_value=16, step=2, default=16), # 8
                activation=swish, name=f'actor_common{n_layer}')(hidden_input)
            hidden_input = latest_hidden
        mu_hidden = Dense(num_outputs * hp.Int('mu_hidden_units', min_value=2, max_value=16, step=2, default=16), # 4
                          activation=swish, name='mu_hidden')(latest_hidden)

        # get means bound near the observation space
        buy_mu = Dense(num_outputs, activation=swish, name='mu_buys')(mu_hidden)
        sell_mu = Dense(num_outputs, activation=swish, name='mu_sells')(mu_hidden)
        unscaled_mu = tf.where(buy_mu > sell_mu, buy_mu,
                               - sell_mu)  # choose bias based on absolute value of buy vs sell
        mu = unscaled_mu * tf.where(unscaled_mu > 0, ub, lb)

        # generate lower triangle scale matrix
        z_vec = Dense(num_outputs * (num_outputs - 1) / 2, activation='tanh', name='z_layer')(latest_hidden)
        z_vec_clipped = clip_z(z_vec)
        std_vec = Dense(num_outputs, activation=swish, name='stdev_layer')(latest_hidden)
        l_scale = hp.Float('std_scale', min_value=0.1, max_value=5., sampling='log', default=constants.l_scale)
        std_vec = tfp.math.clip_by_value_preserve_gradient(std_vec, 0.0, 1.0) * (ub - lb) * l_scale

        # shape the vectors into a lower triangular matrix
        if constants.FULL_RANK_COVARIANCE:
           L = gen_cholesky(z_vec_clipped)
        W = tf.linalg.diag(std_vec)  # rescale from L_correlation to L_covariance, units capital
        if constants.FULL_RANK_COVARIANCE:
            L = tf.matmul(W, L)
        else:
            L = W
        outputs = (mu, L)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='actor_fc_network')
        return model

    @staticmethod
    def cash_to_shares(args):
        return tf.math.divide_no_nan(args[0], args[1])

    @staticmethod
    @tf.custom_gradient
    def clip_by_value_reverse_gradient(inputs):
        ub = tf.cast(constants.TANH_UB, tf.float32)
        out = tf.clip_by_value(inputs, -ub, ub)

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


# wrapper to the model for keras_tuner
class HyperTrader(kt.HyperModel):
    def __init__(self, out_shape, close_idx):
        super(HyperTrader, self).__init__()
        self.out_shape = out_shape
        self.close_idx = close_idx

    def build(self, hp):
        model = Trader(self.out_shape, hp, self.close_idx)
        return model
