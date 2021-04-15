# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:01:11 2020

@author: Aapo Kössi
"""
import string

others = {}
i = 0
def add(name, val):
    others[name] = val
    
    

NUM_MODELS = 1
TOTAL_TIME = 20*365
TRAIN_TIME = 0.8
VAL_TIME = 0.1
TEST_TIME = 0.1
INPUT_DAYS = 640
WINDOW_LENGTH = 730
WINDOW_DIFF = 7
GAMMA = 0.9
STARTING_CAPITAL = 50000
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
INIT_DECAY_STEPS = 10000
DECAY_RATE = 1e+2
T_MUL = 2.0
M_MUL = 1.5
LR_ALPHA = 1e-3




    