# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:23:24 2021

@author: Aapo Kössi
"""

import sys
sys.path.append('C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader')
import utils
import tensorflow as tf



    
path = 'C:/Users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader/data/ccm4_processed_clean/train'
ds = tf.data.experimental.load(path)
ds = ds.batch(128)
ds = ds.map(utils.hasher, num_parallel_calls=tf.data.AUTOTUNE).unbatch()

savepath = 'C:/Users/aapok/python_projects/TensorFlow/workspace/trader'
utils.save_train(ds, savepath)