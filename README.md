# A2C_trader
Deep neural network learning to trade a subset of the market.

The learning algorithm is A2C, implemented so that the complete steps involving experience generation and training are traced.
We use a stochastic policy with the output of the network as the parameters of a multivariate normal distribution.
The action space is bound so that the model will never be short shares or resort to buying on margin.

built with:
- tensorflow 2.7.0,
- keras_tuner 1.0.4
- numpy
- pandas
