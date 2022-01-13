# A2C_trader
Deep neural network learning to trade a subset of the market.

The learning algorithm is A2C, implemented so that the complete steps involving experience generation and training are traced.
We use a stochastic policy with the output of the network as the parameters of a multivariate normal distribution.
The action space is bound so that the model will never be short shares or resort to buying on margin.

The learning algorithm is the Advantage Actor Critic, A2C from [this paper](https://arxiv.org/abs/1602.01783), implemented by me in fully Tensorflow graph-compiled (running a minibatch of experience generation, losses, backpropagation) code. The trading environment, built from scratch, is also graph compiled by Tensorflow to optimize runtime, and the Tensorflow data module is used to load the training data efficiently. JIT compilation of the training step was attempted, but occasional loading of new data prevented that, so auto-clustering is used. Keras-tuner is implemented for hyperparameter optimization, but the current model configuration is insufficient to achieve good results. 

Tensorboard is used to visualize the training progress, including validation trajectories of profits compared with the returns of the stocks traded and visualizing the hyperparameter optimization progress.

The data is retrieved and cleaned from Wharton Research Data Services CCM Compustat merged database, which is divided into time series windows of 10 stocks, which the model tries to learn to trade concurrently. The data input to the model is OHLCV data and daily dividend distributions (would probably be optimal to just include the dividends in the price variable, but the original thought was to also give the model explicit knowledge of dividend payouts). The data is normalized over the time series and ordered s.t. the best performing stock in the window up to the simulation date is the first input. The model is also given the sectors of the stocks as dummy variables and it's current holdings. The model outputs a buy/sell action that is then normalized according to the model's current available capital, shares held and the stock prices. Buy-actions are scaled down so that the model doesn't buy on margin. A simple convolutional network is implemented as the backbone network to extract features from the time series, after which fc-layers comprise the actor and critic networks. As an alternative to the convnet a multilayer LSTM is implemented as an option for the tuner to choose from. 

The model implemented is very simple and unable to learn the underlying data well enough to beat an index fund. Development has been halted for the forseeable future as I reached my learning goals in implementing hyperparameter optimization and a stable model.

Disclaimer:
This project was originally intended to be private because of the possibilities of use in real-time trading, so the usability is not very good. The `main.py` script has some helpful instructions with the `-h` parameter, but the preprocessing data pipeline consists of a specific query to the CCM-compustat merged database and two preprocessing scripts, so I would be surprised if someone manages to run this without additional instructions. If you are interested, however, open an issue or contact me and I will try to help.

built with:
- tensorflow 2.7.0,
- keras_tuner 1.0.4
- numpy
- pandas
