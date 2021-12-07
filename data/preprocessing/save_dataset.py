# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 19:17:34 2021

@author: Aapo Kössi
"""

import sys
sys.path.append('C:/users/aapok/python_projects/TensorFlow/workspace/trader/A2C_trader')
import utils
import argparse

parser = argparse.ArgumentParser(description='preprocess and save a .csv dataset into 2 snapshots. '\
                                 'third script in data preprocessing pipeline, run after preprocess_csv.py')

# parser.add_argument('save_path', help = 'path to the folder used to save model weights as checkpoints', type = str)
parser.add_argument('-d', '--input_dir', help = 'path to the dir including processed csv split into train, eval and test dirs', type=str)

args = parser.parse_args()
path = args.input_dir
arrs = utils.load_ids(path)
date_col = utils.get_data_index(path).get_loc('date')
ds = utils.gen_train(path, arrs[0], date_col)
utils.save_train(ds, path)