"""Train NPS model."""

import argparse, os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/Documents/CONEDEP/')
import numpy as np
import tensorflow as tf

import convnet.models as models
import convnet.data_pipeline as dp
import convnet.config as config


def main(args):
  
  tf.set_random_seed(1234)
  
  cfg = config.Config()
  cfg.batch_size = args.batch_size
  
  N_path = os.path.join(args.dataset,"Noise")
  P_path = os.path.join(args.dataset,"P_tail")
  S_path = os.path.join(args.dataset,"S_tail")
  
  # data pipeline for positive and negative examples
  N_pipeline = dp.DataPipeline(N_path, cfg, 101, 1, 3, True)
  P_pipeline = dp.DataPipeline(P_path, cfg, 101, 1, 3, True)
  S_pipeline = dp.DataPipeline(S_path, cfg, 101, 1, 3, True)
  
  N_samples = {'data': N_pipeline.samples,
              'label': N_pipeline.labels}
  P_samples = {'data': P_pipeline.samples,
              'label': P_pipeline.labels}
  S_samples = {'data': S_pipeline.samples,
              'label': S_pipeline.labels}
  
  samples = {
    "data" : tf.concat([N_samples["data"],
                        P_samples["data"], 
                        S_samples["data"]],  axis=0),
    "label": tf.concat([N_samples["label"],
                        P_samples["label"],
                        S_samples["label"]], axis=0)}
  
  # model
  model = models.get(args.model, samples, cfg, args.checkpoint_dir, is_training=True)
  
  # train loop
  model.train(
      args.learning_rate,
      resume=args.resume,
      summary_step=10) 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='NPSNet')
  parser.add_argument('--checkpoint_dir', type=str, 
                      default='/home/zhouyj/Documents/CONEDEP/output/NPS_depth5')
  parser.add_argument('--dataset', type=str, 
                      default='/home/zhouyj/Documents/CONEDEP/data/train/NPS_frame1')
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--resume', action='store_true')
  parser.set_defaults(resume=False)
  args = parser.parse_args()

  args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model)

  main(args)
