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
    


MIN_DAYS_AVLB = 1000
DEFAULT_TICKERS = 10
NUM_MODELS = 1
TOTAL_TIME = 3740
VAL_TIME = 2*255
TEST_TIME = 2*255
VAL_STEPS = 170
TEST_STEPS = 170
INPUT_DAYS = 180
WINDOW_LENGTH = 255
WINDOW_DIFF = 90
GAMMA = 0.9
STARTING_CAPITAL = 50000
NOISE_RATIO = 0.002
RF = None
SPECIAL_CHARS = string.punctuation
N_STEPS_UPDATE = 10
N_ENVS = 16
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
EP_SHUFFLE_BUF = TOTAL_TIME // WINDOW_DIFF + 1




    