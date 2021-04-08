# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:47:52 2020

@author: Aapo Kössi
"""

import numpy as np
import tensorflow as tf
from object_detection.utils import dataset_util
import os
from lxml import etree as ET
import pandas as pd
from dataclasses import dataclass


@dataclass
class FileData:
    dataset: tf.data.Dataset
    data_index: pd.Index
    ticker_dict: dict
    onehot_cats: pd.DataFrame
    num_tickers: int
     

class Records:
    """
    
    WIP
    
    """
    path = os.getcwd()
    def write_record(file_data, file_name):
        """
        
        writes dataset and the relevant information into tfrecord and accompanying xml files
        dataset will generate tfrecord
        categoriesd will be saved in numpy .npy file
        number of tickers, the ticker names and dataset column names will be saved in .xml file
        
        """
        def serialize(data):
            feature = {
                file_data.data_index[i] : dataset_util.float_list_feature([data[i].numpy()]) \
                for i in range(len(file_data.data_index))
                }
            proto = tf.train.Example(features = tf.train.Features(feature = feature))
            return proto.SerializeToString()
        def tf_serialize(data):
            tf_string = tf.py_function(
                serialize,
                [data],
                tf.string)
            return tf.reshape(tf_string, ())
        def dict_to_xml(root, name, d):
            """

            Parameters
            ----------
            root : root element of elementtree.
            name : the name of the dictionary, will be used as tag.
            d : the dictionary.

            Returns
            -------
            None
            
            Appends the dictionary as a subelement of the root,
            with keys as name attributes and the values as text,
            for each item in the dict.
            

            """
            elem = ET.SubElement(root, name)
            for key, val in d.items():
                child = ET.Element('item')
                child.set('name', key)
                child.text = str(val)
                elem.append(child)
            return
        def arr_to_xml(root, tag, array):
            index = ET.SubElement(root, tag)
            i = 0
            for elem in array:
                sub = ET.SubElement(index, 'element', name = elem)
                sub.text = str(i)
                i += 1
        
        serialized_dataset = file_data.dataset.map(tf_serialize)
        cat_index = file_data.onehot_cats.columns
        
        record_name = "{}/data/{}.tfrecord".format( Records.path , file_name )    
        writer = tf.data.experimental.TFRecordWriter(record_name)
        print('Writing to tfrecord file...')
        writer.write(serialized_dataset)
        print("dataset saved successfully")
        
        np.save('{}/data/{}'.format(Records.path, file_name), file_data.onehot_cats.values)
        
        xml_name = "{}/data/{}.xml".format(Records.path, file_name)
        root = ET.Element("data")
        

        n_tickers = ET.SubElement(root, 'num_tickers')
        n_tickers.text = str(file_data.num_tickers)
        
        dict_to_xml(root, 'ticker_dict', file_data.ticker_dict)
        
        indexes = {'data_index': file_data.data_index, 'cat_index': cat_index}
        
        for key in indexes:
            arr_to_xml(root, key, indexes[key])
        xml_string = ET.tostring(root, pretty_print = True)
        xml_file = open(xml_name, "wb")
        xml_file.write(xml_string)
        return
        
            
        
        
    def read_record(filename):
        """
        
        opens and parses tfrecord into unbatched dataset of stock data,
        retrieves number of tickers, ticker names, sectors and industries
        and data column names from xml file
        retrieves onehot encoded sectors and industries from .npy file
        """
        def retrieve_dicts(dict_names):
            """
            

            Parameters
            ----------
            dict_names : names of the saved dicts

            Returns
            -------
            dicts : list of parsed dictionaries.

            """
            dicts = []
            for name in dict_names:
                d_elem = root.find(name)
                keys = [elem.get('name') for  elem in d_elem.iter() if elem is not d_elem]
                values = [elem.text for elem in d_elem.iter() if elem is not d_elem]
                d = dict(zip(keys, values))
                dicts.append(d)
            return tuple(dicts)
        def retrieve_indexes(index_names):
            indexes = []
            for name in index_names:
                    index = root.find(name)
                    index = list(elem.get('name') for elem in index if elem is not index)
                    index = pd.Index(index)
                    indexes.append(index)
            return tuple(indexes)
        tree = ET.parse("{}/data/{}.xml".format(Records.path, filename))
        root = tree.getroot()
        num_tickers = int(root.find('num_tickers').text)
        dict_names = ['ticker_dict']
        ticker_dict = retrieve_dicts(dict_names)[0]
        index_names = ['data_index', 'cat_index']
        data_index, cat_index = retrieve_indexes(index_names)
        
        npy_name = '{}/data/{}.npy'.format(Records.path, filename)
        onehot_cat_arr = np.load(npy_name)
        encoded_cats = pd.DataFrame(onehot_cat_arr, columns = cat_index)
        
        raw_dataset = tf.data.TFRecordDataset(["{}/data/{}.tfrecord".format(os.getcwd(), filename)])
        feature_description = {data_index[i]: tf.io.FixedLenFeature([], tf.float32, default_value=0.0) for i in range(len(data_index))}
        def _parse_function(example_proto):
            return tf.io.parse_example(example_proto, feature_description)
        dataset = raw_dataset.map(_parse_function)

        # def combine(element):
        #     for i in range(len(index)):
        #         values.append(element[index[i]])
        #     return tf.constant(values)
        # def tf_combine(element):
        #     tf_element = tf.py_function(
        #         combine,
        #         [element],
        #         tf.float32)
        #     return tf.reshape(tf_element, [9])
        dataset = dataset.map(lambda elem: tf.stack([elem[data_index[i]] for i in range(len(data_index))]))
        
        complete_data = FileData(
            dataset,
            data_index,
            ticker_dict,
            encoded_cats,
            num_tickers)
        
        return complete_data
    
    

    
