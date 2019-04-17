import os, time

import numpy as np
import tensorflow as tf

import tflib.model
import tflib.layers as layers

def get(model_name, inputs, config, checkpoint_dir, session, is_training=False):
  """Returns a Model instance by model name.
  Args:
    name:    class name of the model to instantiate.
    samples: samples as returned by a DataPipeline.
    params:  dict of model parameters.
  """
  return globals()[model_name](inputs, config, checkpoint_dir, session, is_training)


class NPSNet(tflib.model.BaseModel):
  
  def __init__(self, inputs, config, checkpoint_dir, session, is_training=False,
               reuse=False):
    self.is_training = is_training
    self.config = config
    super(NPSNet, self).__init__(inputs, 
                                 checkpoint_dir, 
                                 is_training=is_training,
                                 reuse=reuse,
                                 session = session)
    self.sess = session
  
  def _setup_prediction(self):
    self.batch_size = self.inputs['data'].get_shape().as_list()[0]
    
    current_layer = self.inputs['data']
    print current_layer
    c = 64  # number of channels per conv layer
    ksize = 3  # size of the convolution kernel
    depth = 6
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
        tf.contrib.layers.l2_regularizer(self.config.regularization),
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

