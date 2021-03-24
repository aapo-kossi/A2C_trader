# A2C_trader
deep neural network learning to trade a subset of the market


Initialize training by running main.

Has the option to save and read stock data you have previously written.
Scrapes tickers from yahoo finance screener.

The learning algorithm is A2C, implemented with some parts from OpenAI baselines at https://github.com/openai/baselines/blob/tf2/baselines/a2c/
We use a stochastic policy with the output of the network as the parameters of a multivariate normal distribution.
The action space is bound so that the model will never be short shares or resort to buying on margin.
