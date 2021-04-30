# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 19:15:17 2021

@author: Aapo Kössi
"""

import argparse

parser = argparse.ArgumentParser(description='specify whether to load new data or use presaved files')

parser.add_argument('--usefile')
parser.add_argument('--num_tickers', type=int)
parser.add_argument('--save_data')
parser.add_argument('--stop', action='store_true')