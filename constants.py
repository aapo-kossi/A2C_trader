# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:01:11 2020

@author: Aapo Kössi
"""
# this is where the magic numbers live :D
import string

MIN_DAYS_AVLB = [1500, 250, 250]
MIN_AVG_VOL = 5e4
# MIN_MIN_VOL = 5e3
DEFAULT_TICKERS = 10
NUM_MODELS = 1
VAL_TIME = 2*360
TEST_TIME = 2*360
VAL_STEPS = 300
TEST_STEPS = 300
INPUT_DAYS = 90
WINDOW_LENGTH = 250
WINDOW_DIFF = 60
GAMMA = 0.9
STARTING_CAPITAL = 50000
NOISE_RATIO = 0.002
VOL_NOISE_INTENSITY = 10
RF = None
COST_PER_SHARE = 0.01
MIN_P_COST = 0.15 / 100
MIN_COST = 1.0
SPECIAL_CHARS = string.punctuation
N_STEPS_UPDATE = 12
N_ENVS = 16
N_VAL_ENVS = N_ENVS
N_TEST_ENVS = N_ENVS
l_epsilon = 1e-3
l_scale = 1
N = 255
MIN_SWISH = 0.278464542761
TANH_UB = 0.9
INIT_LR = 5e-5
INIT_DECAY_STEPS = 10000
DECAY_RATE = 1e+2
T_MUL = 2.0
M_MUL = 1.5
LR_ALPHA = 1e-3
EP_SHUFFLE_BUF = 256
SPLIT_LABELS = ['train','eval','test']




    