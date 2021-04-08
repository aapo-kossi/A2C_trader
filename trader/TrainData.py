# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:44:36 2020

@author: aapok
"""

from pandas_datareader import data as pdr
import pandas as pd
import datetime
from datetime import timedelta

class TrainData:
    
    def __today():
        date = datetime.date.today()
        date = [date.day, date.month, date.year]
        return date

    def __init__(self, ticker, length, end=__today()):
        end_obj = datetime.date(day = end[0],month = end[1],year = end[2])
        delta_obj = timedelta(days = length)
        start_obj = end_obj - delta_obj
        start = start_obj.strftime("%m/%d/%y")
        end = end_obj.strftime("%m/%d/%y")
        print('Fetching data for {}...'. format(ticker))
        df = pdr.get_data_tiingo(ticker, start = start, end = end)
        df = df.drop(columns=['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume'])
        complete_dates = pd.date_range(start=start,end=end,tz='Etc/GMT')        
        index = pd.MultiIndex.from_product([[ticker],complete_dates],names=['symbol','date'])
        dividends = df['divCash'].reindex(index,fill_value=0)
        split_factor = df['splitFactor'].reindex(index,fill_value=1.0)
        df = df.reindex(index,method='ffill')
        df['splitFactor'] = split_factor
        df['divCash']=dividends
        volumes = df['volume']
        cond = volumes.diff() != 0
        volumes.where(cond, 0, inplace = True)
        df['volume'] = volumes
        df['splitFactor'] = df['splitFactor'].cumprod()
        df['splitFactor'] = df['splitFactor'] / df['splitFactor'].iloc[-1]
        df[['close', 'high', 'low', 'open', 'volume','divCash']] = \
        df[['close', 'high', 'low', 'open', 'volume','divCash']].multiply(df['splitFactor'], axis = 0)

        df.drop(columns = ['splitFactor'], inplace = True)
        self.__data = df.fillna(0)
        self.singular = True
        print('Done.')
        
    def get_data(self):
        return self.__data

    def generate(tickers, t_train):
        return list(map(lambda sym: TrainData(sym, t_train), tickers))

    def combine(multiple):
        dataset = []
        for data in map(lambda x: x.get_data(), multiple):
            dataset.append(data)
        df = pd.concat(dataset)
        df.sort_index(level=1,inplace=True)
        df.reset_index(level=['symbol'],inplace=True)
        df['symbol'] = pd.Categorical(df['symbol'])
        ticker_keys = df.symbol.cat.categories.tolist()
        ticker_keys = dict(zip(ticker_keys, range(len(ticker_keys))))
        df.drop('symbol', axis = 1, inplace = True)
        return df, ticker_keys
        

