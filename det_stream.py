#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : predict_from_stream.py
# Creation Date : 03-12-2016
# Last Modified : 20-12-2017 by Yijian ZHOU
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
""" Detect event and pick the P wave arrival on a stream (mseed/sac) of
continuous recording.
"""
import os, glob, sys, time
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
    if data_size != 3*(int(args.window_size*100+1)):
        return False
    if stream.max()[0]==0.0 or stream.max()[1]==0.0 or stream.max()[2]==0.0:
        return False
    else:
        return True

def win_slide(stream, win_size, step_size, max_win):
    """Returns a list of sliding win of streams"""
    stream_list=[]
    for i in range(max_win):
        ts = stream[0].stats.starttime + (i*step_size)
        stream_list.append(stream.slice(ts, ts+win_size))
    return stream_list

def preprocess_stream(stream):
    stream = stream.detrend('constant') # rmean + rtr in SAC
    return stream.normalize()

def run_det(samples, win_gen, cfg, ckpt, max_windows):
    """
    run DetNet to output an event_list out of the generated windows
    """
    event_list = []
    prob_list = []
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
              to_fetch = [DetNet.layers['class_prediction'],
                          DetNet.layers['class_prob']]     
              if len(win)<3: print 'missing trace!'; continue
              if data_is_complete(win):
                  # preprocess
                  win = preprocess_stream(win)
                  feed_dict = {samples['data']: fetch_window_data(win)}
                  [label, prob] = det_sess.run(to_fetch, feed_dict)
              else: continue

              is_event = label[0] > 0
              if is_event:
                  n_events += 1
                  event_list.append(idx)
                  prob_list.append(prob[0][1])
                  print 'detected events: {} to {} ({}%)'.format(win[0].stats.starttime, 
                                                                 win[0].stats.endtime,
                                                                 round(prob[0][1]*100, 2))

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
    return event_list, prob_list

def raw_pick(win_gen, event_list, prob_list, out_file, sta):
  """
  raw P: first time amp>max(amp)/10
  raw S: max(amp) for E and N chn
  """
  n_events=0
  for i, idx in enumerate(event_list):
      if idx==len(win_gen)-1: print 'unlucky'; continue
      max_ampi = win_gen[idx].max()
      max_ampj = win_gen[idx+1].max()
      if max_ampi[0]==max_ampj[0] and max_ampi[1]==max_ampj[1]\
         and idx+1 in event_list\
         and prob_list[i]<prob_list[i+1]: continue
      else:
         win = win_gen[idx].detrend('constant').\
               filter('bandpass', freqmin = 2., freqmax=40.).normalize()
         if not data_is_complete(win): print 'brocken trace!'; continue
         # raw pick
         i1 = np.where(win[0].data==win.max()[0])[0][0]
         i2 = np.where(win[1].data==win.max()[1])[0][0]
         i3 = np.where(abs(win[2].data)>0.1)[0][0]
         if min(i1, i2)<100:
            print 'wrong pred!'; continue # maybe [0 ... 0]
         
         tp = win[0].stats.starttime + i3/100.0
         ts = win[0].stats.starttime + max(i1,i2)/100.0
         print 'p_time = {}, s_time = {}, s_index = {}'.\
               format(tp.time, ts.time, [i1, i2])
         out_file.write(unicode('{}, {}, {}\n'.format(sta, tp, ts)))
         n_events+=1

  print "Picked {} events".format(n_events)


def main(args):

  stream_files = sorted(glob.glob(args.stream_path+"*"))
  done_file=[]
  output = 'pick.dat'
  if os.path.exists(args.output+output): os.unlink(args.output+output)
  out_file = open(args.output+output,'a')
  for stream_file in stream_files:
    # Load stream
    net, sta, tm, chn = stream_file.split('.')
    if [sta, tm] not in done_file:
      print "+ Loading Stream {}".format(stream_file)
      done_file.append([sta, tm])
      sts_xyz = sorted(glob.glob(args.stream_path + '*%s*%s*'%(sta, tm)))
      if len(sts_xyz)<3: print 'missing trace!'; continue
      stream0 = Stream(traces = [read(sts_xyz[0])[0],
                                 read(sts_xyz[1])[0], 
                                 read(sts_xyz[2])[0]])
      stream = preprocess_stream(stream0).filter('highpass', freq=2.0)

      if args.max_windows is None:
          total_time_in_sec = stream[0].stats.endtime - stream[0].stats.starttime
          max_windows = (total_time_in_sec - args.window_size) / args.window_step
      else:
          max_windows = args.max_windows

      win_gen = win_slide(stream0, args.window_size, args.window_step, max_windows)
      cfg = config.Config(); cfg.batch_size = 1
      samples = {'data': tf.placeholder(tf.float32,
                       shape=(cfg.batch_size, int(args.window_size*100+1), 3),
                       name='input_data')}
      # run det
      event_list, prob_list = run_det(samples, win_gen, cfg, args.det_ckpt, max_windows)
      # output raw pick
      raw_pick(win_gen, event_list, prob_list, out_file, sta)

  out_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_path",type=str,
                        default='/data/WC_mon78/',
                        help="path to mseed to analyze")
    parser.add_argument("--det_ckpt",type=str,
                        default='/home/zhouyj/Documents/CONEDEP/output/WC_man30s_ver1/DetNet',
                        help="path to directory of chekpoints")
    parser.add_argument("--window_size",type=int,default=30.0,
                        help="size of the window to analyze")
    parser.add_argument("--window_step",type=int,default=15.0,
                        help="step between windows to analyze")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to analyze")
    parser.add_argument("--output",type=str,
                        default="/home/zhouyj/Documents/CONEDEP/output/WC_man30s/picks/",
                        help="dir of predicted events")
    args = parser.parse_args()
    main(args)

