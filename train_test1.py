import argparse, os, time, sys
sys.path.append('/home/zhouyj/Documents/CONEDEP/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

import test1 as models
from convnet.data_pipeline import DataPipeline
import convnet.config as config


def main(args):
    tf.set_random_seed(1234)
    
    cfg = config.Config()
    cfg.batch_size = args.batch_size
    
    samples = {
        'data': DataPipeline(args.dataset, cfg, 3001, True).samples,
        'ppk' : DataPipeline(args.dataset, cfg, 3001, True).ppks
        }
    
    with tf.Session() as sess:
      # model
      model = models.get('PpkNet', samples, cfg, args.checkpoint_dir, sess, is_training=True)
      
      # train loop
      model.train(
        args.learning_rate,
        resume=args.resume,
        profiling=args.profiling,
        summary_step=10) 
    
if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--checkpoint_dir', type=str, 
                      default='/home/zhouyj/Documents/CONEDEP/output/WC_man30s_snr10tsp10')
   parser.add_argument('--dataset', type=str, 
                      default='/home/zhouyj/Documents/CONEDEP/data/train/WC_man30s/snr10_tsp10/positive')
   parser.add_argument('--batch_size', type=int, default=64)
   parser.add_argument('--learning_rate', type=float, default=1e-3)
   parser.add_argument('--resume', action='store_true')
   parser.set_defaults(resume=False)
   parser.add_argument('--profiling', action='store_true')
   parser.set_defaults(profiling=False)
   args = parser.parse_args()
   
   args.checkpoint_dir = os.path.join(args.checkpoint_dir, 'PpkNet')
   main(args)
