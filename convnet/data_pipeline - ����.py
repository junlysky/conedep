"""Classes and functions to read, write and feed data."""
import os
import numpy as np
import tensorflow as tf


class DataWriter(object):
    """ 
    Writes .tfrecords file to disk from window np.array objects.
    """
    
    def __init__(self, filename):
        self._filename = filename
        self._written = 0
        self._writer = tf.python_io.TFRecordWriter(self._filename)
    
    def write(self, data, label, p_time=-1., s_time=-1.):
        # get shape (num_step * step_len * chns)
        n_steps   = data.shape[0]
        n_traces  = data.shape[1]
        n_samples = data.shape[2]
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'window_size': self._int64_feature(n_samples),
            'n_traces':    self._int64_feature(n_traces),
            'n_steps':     self._int64_feature(n_steps),
            'data':        self._bytes_feature(data.tobytes()),
            'label':       self._int64_feature(label),
            'p_time':      self._float_feature(p_time),
            's_time':      self._float_feature(s_time)
        }))
        self._writer.write(example.SerializeToString())
        self._written += 1
    
    def close(self):
        self._writer.close()
    
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class DataReader(object):
    
    def __init__(self, path, config, win_len, n_steps, n_traces, shuffle=True):
        self._path    = path
        self._shuffle = shuffle
        self._config  = config
        self.win_size = win_len #TODO
        self.n_traces = n_traces
        self.n_steps  = n_steps
        self._reader  = tf.TFRecordReader()
    
    def read(self):
        filename_queue = self._filename_queue()
        _, serialized_example = self._reader.read(filename_queue)
        example = self._parse_example(serialized_example)
        return example
    
    def _filename_queue(self):
        fnames = []
        for root, dirs, files in os.walk(self._path):
            for f in files:
                if f.endswith(".tfrecords"):
                   fnames.append(os.path.join(root, f))
        fname_q = tf.train.string_input_producer(fnames,
                                                 shuffle=self._shuffle,
                                                 num_epochs=self._config.n_epochs)
        return fname_q
    
    def _parse_example(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'window_size': tf.FixedLenFeature([], tf.int64),
                'n_traces':    tf.FixedLenFeature([], tf.int64),
                'n_steps':     tf.FixedLenFeature([], tf.int64),
                'data':        tf.FixedLenFeature([], tf.string),
                'label':       tf.FixedLenFeature([], tf.int64),
                'p_time':      tf.FixedLenFeature([], tf.float32),
                's_time':      tf.FixedLenFeature([], tf.float32),
                })
        
        # Convert and reshape
        data = tf.decode_raw(features['data'], tf.float32)
        data.set_shape([self.n_steps * self.n_traces * self.win_size])
        data = tf.reshape(data, [self.n_steps, self.n_traces, self.win_size])
        data = tf.transpose(data, [0, 2, 1])
        
        # Pack
        features['data'] = data
        return features


class DataPipeline(object):
    """Creates a queue op to stream data for training.
    Attributes:
    samples: Tensor(float). batch of input samples [batch_size, n_channels, n_points]
    labels:  Tensor(int32). Corresponding batch 0 or 1 labels, [batch_size,]
    """
    
    def __init__(self, dataset_path, config, win_len, n_steps, n_traces, is_training):
        
        min_after_dequeue = 1000
        capacity = 1000 + 3 * config.batch_size #TODO
        
        if is_training:
            with tf.name_scope('inputs'):
                self._reader = DataReader(dataset_path, config=config, win_len=win_len, n_steps=n_steps, n_traces=n_traces)
                samples      = self._reader.read()
                sample_input = samples["data"]
                sample_label = samples["label"] 
                sample_ppk   = [samples["p_time"], samples["s_time"]]
                
                self.samples, self.labels, self.ppks = tf.train.shuffle_batch(
                    [sample_input, sample_label, sample_ppk],
                    batch_size=config.batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    allow_smaller_final_batch=False)
        
        elif not is_training:
            with tf.name_scope('validation_inputs'):
                self._reader = DataReader(dataset_path, config=config, win_len=win_len, n_steps=n_steps, n_traces=n_traces)
                samples      = self._reader.read()
                sample_input = samples["data"]
                sample_label = samples["label"], 
                sample_ppk   =  [samples["p_time"], samples["s_time"]]
                
                self.samples, self.labels, self.ppks = tf.train.batch(
                    [sample_input, sample_label, sample_ppk],
                    batch_size=config.batch_size,
                    capacity=capacity,
                    num_threads=config.n_threads,
                    allow_smaller_final_batch=False)
        else:
            raise ValueError(
                "is_training flag is not defined, set True for training and False for testing")
