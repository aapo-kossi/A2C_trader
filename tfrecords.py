# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:17:59 2021

@author: Aapo Kössi
"""

import tensorflow as tf
from object_detection.utils import dataset_util
import glob

def write_record(ds, path, shards = 10, elems_per_record = 16384):
    def serialize(*data):
        feature = {
            f'feature_{i}' : dataset_util.bytes_list_feature([tf.io.serialize_tensor(data[i]).numpy()]) \
            for i in range(len(data))
            }
        proto = tf.train.Example(features = tf.train.Features(feature = feature))
        return proto.SerializeToString()
    def tf_serialize(*data):
        tf_string = tf.py_function(
            serialize,
            data,
            tf.string)
        return tf.reshape(tf_string, ())
    
    
    serialized_dataset = ds.map(tf_serialize)

    for n in range(shards):
        shard = serialized_dataset.shard(shards, n).take(elems_per_record)
        record_name = f'{path}/{n}.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(record_name)
        print('Writing to tfrecord file...')
        writer.write(shard)
        print("dataset saved successfully")
    
    
def read_record(path, batch_size = 64):
    
    feature_desc = {
            'feature_0': tf.io.FixedLenFeature((), tf.string),
            'feature_1': tf.io.FixedLenFeature((), tf.string),
            'feature_2': tf.io.FixedLenFeature((), tf.string)
            }
    
    def parse_bytes(bytes_elem):
        return tf.io.parse_example(bytes_elem, feature_desc)
        
    def parse_features(examples):
        def map_func(x): tf.io.parse_tensor(x, tf.float32)
        elem = tuple(examples.values())
        elem = tuple(map(map_func, elem))
        #tf.ensure_shape(elem[0], ...)
        #tf.ensure...
        #...
        return elem
    
    pattern = f'{path}/*.tfrecord'
    files = glob.glob(pattern)
    num_shards = len(files)
    ds = tf.data.TFRecordDataset(pattern, num_parallel_reads = num_shards)
    ds = ds.batch(batch_size).map(parse_bytes, num_parallel_calls = tf.data.AUTOTUNE)
    ds = ds.map(parse_features, num_parallel_calls = tf.data.AUTOTUNE)
    ds = ds.unbatch()   
    return ds


