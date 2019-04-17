# -------------------------------------------------------------------
# File Name : models.py
# Creation Date : 11-27-16
# Last Modified : 2017-09-16 by Yijian Zhou
# Author: Thibaut Perol & Michael Gharbi <tperol@g.harvard.edu>
# -------------------------------------------------------------------

import os, time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import tflib.model
import tflib.layers as layers
#from tflib.yellowfin import *

def get(model_name, inputs, config, checkpoint_dir, is_training=False):
  """Returns a Model instance by model name.
  Args:
    name:    class name of the model to instantiate.
    samples: samples as returned by a DataPipeline.
    params:  dict of model parameters.
  """
  return globals()[model_name](inputs, config, checkpoint_dir, is_training)


class DetNet(tflib.model.BaseModel):
  
  def __init__(self, inputs, config, checkpoint_dir, is_training=False,
               reuse=False):
    self.is_training = is_training
    self.config = config
    super(DetNet, self).__init__(inputs, 
                                 checkpoint_dir, 
                                 is_training=is_training,
                                 reuse=reuse)
  
  def _setup_prediction(self):
    # def inputs
    self.batch_size = self.inputs['data'].get_shape().as_list()[0]
    current_layer = self.inputs['data']
    # def hypo-params
    c     = 64 # number of channels per conv layer
    ksize = 3  # size of the convolution kernel
    depth = 10 
    # def model structure
    for i in range(depth):
        current_layer = layers.conv1(current_layer, c, ksize, stride=1, 
                                     scope='conv{}'.format(i+1), padding='SAME')
        current_layer = layers.max_pooling(current_layer, 2)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
        self.layers['conv{}'.format(i+1)] = current_layer
    # softmax regression
    bs, width, _ = current_layer.get_shape().as_list()
    current_layer = tf.reshape(current_layer, [bs, width*c], name="reshape")
    
    current_layer = layers.fc(current_layer, 2, scope='det_logits', activation_fn=None)
    self.layers['logits'] = current_layer
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
    # output prediction
    self.layers['class_prob'] = tf.nn.softmax(current_layer, name='det_class_prob')
    self.layers['class_prediction'] = tf.argmax(self.layers['class_prob'], 1, name='det_class_pred')
    
    # add L2 regularization
    tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(1e-3),
        weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))
  
  # toprint validation messages
  def validation_metrics(self):
    if not hasattr(self, '_validation_metrics'):
      self._setup_loss()
      
      self._validation_metrics = {
        'loss': self.loss,
        'detection_accuracy': self.detection_accuracy
      }
    return self._validation_metrics
  
  def validation_metrics_message(self, metrics):
    s = 'loss = {:.5f} | det. acc. = {:.1f}%'.format(metrics['loss'], \
         metrics['detection_accuracy']*100)
    return s
  
  # def loss function and accuracy
  def _setup_loss(self):
    with tf.name_scope('loss'):
      targets = self.inputs['label']
      raw_loss = tf.reduce_mean(
                   tf.nn.sparse_softmax_cross_entropy_with_logits(
                     logits = self.layers['logits'], labels = targets))
      self.summaries.append(tf.summary.scalar('loss/train_raw', raw_loss))
    
    self.loss = raw_loss
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if reg_losses:
      with tf.name_scope('regularizers'):
        reg_loss = tf.reduce_sum(reg_losses)
        self.summaries.append(tf.summary.scalar('loss/regularization', reg_loss))
      self.loss += reg_loss
    
    self.summaries.append(tf.summary.scalar('loss/train', self.loss))
    
    with tf.name_scope('accuracy'):
      is_true_event = tf.cast(tf.greater(targets, tf.zeros_like(targets)), tf.int64)
      is_pred_event = tf.cast(tf.greater(self.layers['class_prediction'], 
                                         tf.zeros_like(targets)), tf.int64)
      detection_is_correct = tf.equal(is_true_event, is_pred_event)
      self.detection_accuracy = tf.reduce_mean(tf.to_float(detection_is_correct))
      self.summaries.append(tf.summary.scalar('detection_accuracy/train', self.detection_accuracy))
  
  # def optimizer
  def _setup_optimizer(self, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops, name='update_ops')
      with tf.control_dependencies([updates]):
        self.loss = tf.identity(self.loss)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(
        self.loss, name='optimizer', global_step=self.global_step)
    self.optimizer = optim
  
  # def train messages
  def _tofetch(self):
    return {
        'optimizer': self.optimizer,
        'loss': self.loss,
        'detection_accuracy': self.detection_accuracy,
    }
  
  def _summary_step(self, step_data):
    step = step_data['step']
    loss = step_data['loss']
    det_accuracy = step_data['detection_accuracy']
    duration = step_data['duration']
    avg_duration = 1000*duration/step
    
    if self.is_training:
      toprint ='Step {} | {:.0f}s ({:.0f}ms) | loss = {:.4f} | det. acc. = {:.1f}%'.format(
        step, duration, avg_duration, loss, 100*det_accuracy)
    else:
      toprint ='Step {} | {:.0f}s ({:.0f}ms) | accuracy = {:.1f}%'.format(
        step, duration, avg_duration, 100*det_accuracy)
    
    return toprint



class PpkNet(tflib.model.BaseModel):
  
  def __init__(self, inputs, config, checkpoint_dir, 
               is_training=False, reuse=False):
    self.is_training = is_training
    self.config = config
    super(PpkNet, self).__init__(inputs, 
                                 checkpoint_dir, 
                                 is_training = is_training, 
                                 reuse = reuse) 
  
  def _setup_prediction(self):
  
    # RNN params
    self.num_units  = 50  # number of gru cells per layer
    self.num_layers = 2   # number of gru layers
    
    # input data
    self.batch_size = self.inputs['data'].get_shape().as_list()[0]
    current_layer   = self.inputs['data']
    bs, self.num_step, step_len, chn = current_layer.get_shape().as_list()
    print 'models/PpkNet.input', current_layer
    current_layer = tf.reshape(current_layer, [bs, self.num_step, -1]) # flatten the 3 chns
    # input label
    tps = self.inputs['ppk']
    step_stride = 50 #TODO
    self.label = tf.to_int32(100*tps/step_stride) -1
    
    # RNN PpkNet
#    output, state = layers.gru(data, num_layers=self.num_layers, num_units=self.num_units, 
#                               batch_size=self.batch_size, scope='multi_gru')
    output0, state = layers.bi_gru(current_layer, 
                                   num_layers=self.num_layers, 
                                   num_units =self.num_units,
                                   batch_size=self.batch_size, 
                                   scope='bi_gru')
    output = tf.concat([output0[0], output0[1]], axis=2)
#    output, state = layers.lstm(data, self.num_layers, num_units=self.num_units,
#                                batch_size=self.batch_size, scope='multi_lstm')
#    output, state = layers.bi_lstm(data, num_layers=self.num_layers, num_units=self.num_units,
#                                   batch_size=self.batch_size, scope='bi_lstm')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
    self.layers['rnn'] = output
    
    output = tf.reshape(output, [-1, 2*self.num_units]) # flatten
    logits = layers.fc(output, 3, scope='ppk_logits', activation_fn=None) # [0 1 2] for [before P, before S, after S] 
    self.layers['logits'] = logits
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, logits)
    
    flat_pred = tf.nn.softmax(logits, name='ppk_class_prob')
    self.layers['class_prob'] = tf.reshape(flat_pred, [-1, self.num_step, 3]) # shape=(32, 59, 3)
    self.layers['class_prediction'] = tf.argmax(self.layers['class_prob'], 2, name='ppk_class_pred')
  
  
  def validation_metrics(self):
    if not hasattr(self, '_validation_metrics'):
         self._setup_loss()
         self._validation_metrics = {'loss': self.loss,
                                     'err_rate': self.err_rate}
    return self._validation_metrics
  
  def validation_metrics_message(self, metrics):
      s = 'loss = {:.5f} | det. acc. = {:.1f}'.format(metrics['loss'],
          metrics['err_rate'])
      return s

  def _setup_loss(self):
    with tf.name_scope('loss'):
      index_ps = self.label
      # one-hot target
      print 'models/setup_loss'
      for batchi in range(self.batch_size):
          indexi = index_ps[batchi]
          
          target_o = tf.concat([tf.ones( [indexi[0],1]), 
                                tf.zeros([indexi[0],1]),
                                tf.zeros([indexi[0],1])], axis=1)
          
          target_p = tf.concat([tf.zeros([indexi[1] - indexi[0], 1]), 
                                tf.ones( [indexi[1] - indexi[0], 1]),
                                tf.zeros([indexi[1] - indexi[0], 1])], axis=1)
          
          target_s = tf.concat([tf.zeros([self.num_step - indexi[1], 1]), 
                                tf.zeros([self.num_step - indexi[1], 1]),
                                tf.ones( [self.num_step - indexi[1], 1])], axis=1)
#          to_mask  = tf.zeros()
          targeti = tf.concat([target_o, target_p, target_s], axis=0)
          targeti = tf.expand_dims(targeti, axis=0)
          if batchi==0: target = targeti
          else:         target = tf.concat([target, targeti], axis=0)
      
      self.targets = tf.argmax(target, 2)
#      self.mask = tf.reduce_max(target, 2)
      flat_target = tf.reshape(self.targets, [-1])
#      flat_mask = tf.reshape(self.mask, [-1])
      
      # cross entropy loss
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                            logits = self.layers['logits'], 
                            labels = flat_target)
      loss = tf.reduce_sum(cross_entropy)
      
      self.summaries.append(tf.summary.scalar('/train', loss))
      self.loss = loss
    
    with tf.name_scope('err_rate'):
      ppk_is_incorrect = tf.to_float(tf.not_equal(self.targets, 
                                                  self.layers['class_prediction']))
      self.err_rate = tf.reduce_sum(ppk_is_incorrect)/self.batch_size
      self.summaries.append(tf.summary.scalar('err_rate/train', self.err_rate))
  
  def _setup_optimizer(self, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
       updates = tf.group(*update_ops, name='update_ops')
       with tf.control_dependencies([updates]):
            self.loss = tf.identity(self.loss)
    optim = tf.train.AdamOptimizer(learning_rate)
#    optim = tf.train.MomentumOptimizer(learning_rate, 0.9)
#    optim = YFOptimizer()
    self.optimizer = optim.minimize(self.loss, 
                                    name='optimizer', 
                                    global_step=self.global_step)
  
  def _tofetch(self):
    return { 'optimizer': self.optimizer,
             'loss': self.loss,
             'err_rate': self.err_rate,
             'target': self.targets,
             'pred': self.layers['class_prediction'],
             'label': self.inputs['ppk']
           }
  
  def _summary_step(self, step_data):
    step = step_data['step']
    loss = step_data['loss']
    err_rate = step_data['err_rate']
    duration = step_data['duration']
    avg_duration = 1000*duration /step
    
    if self.is_training:
      toprint ='Step {} | {:.0f}s ({:.0f}ms) | loss = {:.4f} | err_rate = {:.1f}'.\
                format(step, duration, avg_duration, loss, err_rate)
    else:
      toprint = 'Step {} | {:.0f}s ({:.0f}ms) | err_rate = {:.1f}'.\
                format(step, duration, avg_duration, err_rate)
    return toprint



class NPSNet(tflib.model.BaseModel):
  
  def __init__(self, inputs, config, checkpoint_dir, is_training=False,
               reuse=False):
    self.is_training = is_training
    self.config = config
    super(NPSNet, self).__init__(inputs, 
                                 checkpoint_dir, 
                                 is_training=is_training,
                                 reuse=reuse)
    
  def _setup_prediction(self):
    self.batch_size = self.inputs['data'].get_shape().as_list()[0]
    current_layer = self.inputs['data']
    print 'models.NPSNet', current_layer
    current_layer = tf.squeeze(current_layer, [1])
    
    c = 64  # number of channels per conv layer
    ksize = 3  # size of the convolution kernel
    depth = 5
    
    # def network structure
    for i in range(depth):
        current_layer = layers.conv1(current_layer, c, ksize, stride=1, 
                                     scope='conv{}'.format(i+1), padding='SAME')
        current_layer = layers.max_pooling(current_layer, 2)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
        self.layers['conv{}'.format(i+1)] = current_layer
    
    bs, width, _ = current_layer.get_shape().as_list()
    current_layer = tf.reshape(current_layer, [bs, width*c], name="reshape")
    
    current_layer = layers.fc(current_layer, 3, scope='det_logits', activation_fn=None)
    self.layers['logits'] = current_layer
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
    
    self.layers['class_prob'] = tf.nn.softmax(current_layer, name='det_class_prob')
    self.layers['class_prediction'] = tf.argmax(self.layers['class_prob'], 1, name='det_class_pred')
    
    tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(1e-3),
        weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))
  
  
  def validation_metrics(self):
    if not hasattr(self, '_validation_metrics'):
      self._setup_loss()
      
      self._validation_metrics = {
        'loss': self.loss,
        'detection_accuracy': self.detection_accuracy
      }
    return self._validation_metrics
  
  def validation_metrics_message(self, metrics):
    s = 'loss = {:.5f} | det. acc. = {:.1f}%'.format(metrics['loss'], \
         metrics['detection_accuracy']*100)
    return s
  
  def _setup_loss(self):
    with tf.name_scope('loss'):
      targets = self.inputs['label']
      raw_loss = tf.reduce_mean(
                 tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = self.layers['logits'], labels = targets))
      self.summaries.append(tf.summary.scalar('loss/train_raw', raw_loss))
    
    self.loss = raw_loss
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if reg_losses:
      with tf.name_scope('regularizers'):
        reg_loss = tf.reduce_sum(reg_losses)
        self.summaries.append(tf.summary.scalar('loss/regularization', reg_loss))
      self.loss += reg_loss
    
    self.summaries.append(tf.summary.scalar('loss/train', self.loss))
    
    with tf.name_scope('accuracy'):
      is_true_event = tf.cast(tf.greater(targets, tf.zeros_like(targets)), tf.int64)
      is_pred_event = tf.cast(tf.greater(self.layers['class_prediction'], 
                                         tf.zeros_like(targets)), tf.int64)
      detection_is_correct = tf.equal(is_true_event, is_pred_event)
      self.detection_accuracy = tf.reduce_mean(tf.to_float(detection_is_correct))
      self.summaries.append(tf.summary.scalar('detection_accuracy/train', self.detection_accuracy))
  
  def _setup_optimizer(self, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops, name='update_ops')
      with tf.control_dependencies([updates]):
        self.loss = tf.identity(self.loss)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(
        self.loss, name='optimizer', global_step=self.global_step)
    self.optimizer = optim
  
  def _tofetch(self):
    return {
        'optimizer': self.optimizer,
        'loss': self.loss,
        'detection_accuracy': self.detection_accuracy
    }
  
  def _summary_step(self, step_data):
    step = step_data['step']
    loss = step_data['loss']
    det_accuracy = step_data['detection_accuracy']
    duration = step_data['duration']
    avg_duration = 1000*duration/step
    
    if self.is_training:
      toprint ='Step {} | {:.0f}s ({:.0f}ms) | loss = {:.4f} | det. acc. = {:.1f}%'.format(
        step, duration, avg_duration, loss, 100*det_accuracy)
    else:
      toprint ='Step {} | {:.0f}s ({:.0f}ms) | accuracy = {:.1f}%'.format(
        step, duration, avg_duration, 100*det_accuracy)
    
    return toprint
