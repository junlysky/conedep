import argparse, os, time, sys
sys.path.append('/home/zhouyj/Documents/CONEDEP/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

import convnet.models as models
from convnet.data_pipeline import DataPipeline
import convnet.config as config


tf.set_random_seed(1234)

cfg = config.Config()
cfg.batch_size = 64

dataset = '/home/zhouyj/Documents/CONEDEP/data/train/WC_man30s/all/ppk_depth5'
samples = {
        'data': DataPipeline(dataset, cfg, 4, 59, 64, True).samples,
        'ppk' : DataPipeline(dataset, cfg, 4, 59, 64, True).ppks
        }

# model
checkpoint_dir = '/home/zhouyj/Documents/CONEDEP/output/WC_man30s_all'
model = models.get('PpkNet', samples, cfg, checkpoint_dir, is_training=True)

print model


sess = tf.Session()
sess.run(model.label)



# train loop
#model.train(1e-3, resume=False, summary_step=10) 
# unfold:

lr = tf.Variable(learning_rate, name='learning_rate',
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES])

# Optimizer
#self._setup_loss()
#self._setup_optimizer(lr)

model._setup_loss()
model._setup_optimizer(lr)

run_options = None
run_metadata = None

# Summaries
#self.merged_summaries = tf.summary.merge(self.summaries)

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
    
    print 'Initializing all variables.'
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    
    print 'Starting data threads coordinator.'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    print 'Starting optimization.'
    start_time = time.time()
    step=0
      try:
        while not coord.should_stop():  # Training loop
          step_data = self._train_step(sess, start_time, run_options, run_metadata)
          step = step_data['step']
          
          if step > 0 and step % summary_step == 0:
#            print('target=', step_data['target'][0:10])
#            print('pred = ', step_data['pred'][0:10])
            np.set_printoptions(threshold='nan') 
            print self._summary_step(step_data)
            self.summary_writer.add_summary(step_data['summaries'], global_step=step)

          # Save checkpoint every `checkpoint_step`
          if checkpoint_step is not None and (
              step > 0) and step % checkpoint_step == 0:
            print 'Step {} | Saving checkpoint.'.format(step)
            self.save(sess)
      

