# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:01:11 2020

@author: Aapo Kössi
"""
import string

others= {}
i = 0
def add(name, val):
    others[name] = val
    


MIN_DAYS_AVLB = 250
MIN_AVG_VOL = 1e5
# MIN_MIN_VOL = 5e3
DEFAULT_TICKERS = 10
NUM_MODELS = 1
VAL_TIME = 2*360
TEST_TIME = 2*360
VAL_STEPS = 200
TEST_STEPS = 200
INPUT_DAYS = 90
WINDOW_LENGTH = 150
WINDOW_DIFF = 60
# MAP_BATCH_SIZE = 512           # for vectorized mapping of datasets, didn't end up happening
GAMMA = 0.9
STARTING_CAPITAL = 50000
NOISE_RATIO = 0.002
RF = None
SPECIAL_CHARS = string.punctuation
N_STEPS_UPDATE = 10
N_ENVS = 16
N_VAL_ENVS = N_ENVS
N_TEST_ENVS = N_ENVS
l_epsilon = 1e-3
l_scale = 1
N = 255
MIN_SWISH = 0.278464542761
TANH_UB = 0.9
INIT_LR = 1e-3
INIT_DECAY_STEPS = 100000
DECAY_RATE = 1e+2
T_MUL = 2.0
M_MUL = 1.5
LR_ALPHA = 1e-3
EP_SHUFFLE_BUF = 256
CORRUPT_CONAMES = ['TRAVEL + LEISURE CO', 'COMMCAST']




    