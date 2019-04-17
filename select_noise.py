#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : predict_from_stream.py
# Creation Date : 03-12-2016
# Last Modified : 19-10-2017 by Yijian ZHOU
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
""" Detect event and pick the P wave arrival on a stream (mseed/sac) of
continuous recording.
"""
import os, glob, sys, shutil, time
sys.path.append('/home/zhouyj/Documents/CONEDEP/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse

import numpy as np
import tensorflow as tf
from obspy.core import *

import convnet.models as models
from convnet.data_pipeline import DataPipeline
import convnet.config as config

def fetch_window_data(stream):
    """fetch data from a stream window and dump in np array"""
    data = np.empty((int(args.window_size*100+1), 3))
    for i in range(3):
        data[:, i] = stream[i].data.astype(np.float32)
    data = np.expand_dims(data, 0)
    return data

def data_is_complete(stream):
    """Returns True if there is 1001*3 points in win"""
    data_size = len(stream[0].data) + len(stream[1].data) + len(stream[2].data)
    if data_size == 3*(int(args.window_size*100+1)):
        return True
    else:
        return False

def preprocess_stream(stream):
    stream = stream.detrend('constant') # rmean + rtr in SAC
    return stream.normalize()

def run_det(samples, win_gen, cfg, ckpt, max_windows, sta):
    """
    run DetNet to output an event_list out of the generated windows
    """
    event_list = []
    with tf.Session() as det_sess:
        # set up DetNet model and validation metrics
        DetNet = models.get('DetNet', samples, cfg, ckpt)
        DetNet.load(det_sess, None)
        print 'Predicting using model at step {}'.format(
               det_sess.run(DetNet.global_step))

        n_events = 0
        time_start = time.time()
        try:
            for idx, win in enumerate(win_gen):
                # Fetch label
                to_fetch = DetNet.layers['class_prediction']
                if len(win)<3: print 'missing trace!'; continue
                if data_is_complete(win):
                    # preprocess
                    win = preprocess_stream(win)
                    feed_dict = {samples['data']: fetch_window_data(win)}
                    label = det_sess.run(to_fetch, feed_dict)
                else:
                    continue

                is_event = label[0] > 0
                if is_event: n_events += 1

                if is_event:
                    event_list.append(idx)
                    if win[0].max()==0.0 or win[1].max()==0.0 or win[2].max()==0.0: print 'broken trace!'; continue
                    print 'detected events: {} to {}'.format(win[0].stats.starttime, win[0].stats.endtime)
                    t0 = win[0].stats.starttime
                    Stream(traces=[win[0]]).write(args.output + 
                           '.'.join([sta, str(t0.year)+str(t0.julday)+
                           str(t0.hour).zfill(2) + str(t0.minute).zfill(2), 'BHE', 'SAC']))
                    Stream(traces=[win[1]]).write(args.output +
                           '.'.join([sta, str(t0.year)+str(t0.julday)+
                           str(t0.hour).zfill(2 )+ str(t0.minute).zfill(2), 'BHN', 'SAC']))
                    Stream(traces=[win[2]]).write(args.output +
                           '.'.join([sta, str(t0.year)+str(t0.julday)+
                           str(t0.hour).zfill(2) + str(t0.minute).zfill(2), 'BHZ', 'SAC']))


                if idx % 1000 ==0:
                    print "Analyzing {} records".format(win[0].stats.starttime)

                if idx >= max_windows:
                    print "stopped after {} windows".format(max_windows)
                    print "found {} events".format(n_events)
                    break

        except KeyboardInterrupt:
            print 'Interrupted at time {}.'.format(win[0].stats.starttime)
            print "processed {} windows, found {} events".format(idx+1,n_events)
            print "Run time: ", time.time() - time_start

    print "DetNet Run time: ", time.time() - time_start
    print "found {} events".format(n_events)
    tf.reset_default_graph()
    return event_list


def main(args):

  stream_files = sorted(glob.glob(args.stream_path+"*2008221*"))
  done_file=[]
  for stream_file in stream_files:
    # Load stream
    net, sta, tm, chn = stream_file.split('.')
    if [sta, tm] not in done_file:
      print "+ Loading Stream {}".format(stream_file)
      done_file.append([sta, tm])
      sts_xyz = sorted(glob.glob(args.stream_path + '*%s*%s*'%(sta, tm)))
      if len(sts_xyz)<3: print 'missing trace!'; continue
      stream = Stream(traces = [read(sts_xyz[0])[0], read(sts_xyz[1])[0], read(sts_xyz[2])[0]])

      # Windows generator
      win_gen = stream.slide(window_length=args.window_size,
                           step=args.window_step)
      if args.max_windows is None:
          total_time_in_sec = stream[0].stats.endtime - stream[0].stats.starttime
          max_windows = (total_time_in_sec - args.window_size) / args.window_step
      else:
          max_windows = args.max_windows

      # run det
      cfg = config.Config(); cfg.batch_size = 1
      samples = {'data': tf.placeholder(tf.float32,
                       shape=(cfg.batch_size, int(args.window_size*100+1), 3),
                       name='input_data'),
                 'ppk': tf.placeholder(tf.float32,
                        shape=(cfg.batch_size,2),
                        name='input_ppk')}
      event_list = run_det(samples, win_gen, cfg, args.det_ckpt, max_windows, sta)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_path",type=str,
                        default='/data/WC_mon78/',
                        help="path to mseed to analyze")
    parser.add_argument("--det_ckpt",type=str,
                        default='/home/zhouyj/Documents/CONEDEP/output/WC_man30s/DetNet',
                        help="path to directory of chekpoints")
    parser.add_argument("--window_size",type=int,default=30.0,
                        help="size of the window to analyze")
    parser.add_argument("--window_step",type=int,default=40.0,
                        help="step between windows to analyze")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to analyze")
    parser.add_argument("--output",type=str,
                        default="/data/WC_AItrain/Man_30s/Noise_aug/",
                        help="dir of predicted events")
    args = parser.parse_args()
    main(args)

