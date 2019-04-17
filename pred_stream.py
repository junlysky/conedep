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
import convnet.config as config
from convnet.data_pipeline import DataPipeline

def fetch_window_data(stream, num_step, step_len, step_stride):
    """fetch data from a stream window and dump in np array"""
    # convert to numpy array
    xdata = stream[0].data
    ydata = stream[1].data
    zdata = stream[2].data
    st_data = np.array([xdata, ydata, zdata])
    # feed into time steps
    time_steps = np.zeros((1, num_step, step_len+1, 3), dtype=np.float32)
    for i in range(num_step):
        idx_s = i * step_stride
        idx_e = idx_s +step_len +1
        current_step = st_data[:, idx_s:idx_e]
        if current_step.shape[1] == step_len+1: 
            time_steps[0, i, :, :] = np.transpose(current_step)
    
    return time_steps


def win_slide(stream, start_time, win_size, step_size, max_windows):
    """Returns a list of sliding win of streams"""
    stream_list=[]
    for i in range(max_windows):
        ts = start_time + (i*step_size)
        st = stream.slice(ts, ts+win_size)
        # skip missing data
        if len(st)!=3: continue
        if not st[0].stats.starttime == st[1].stats.starttime == st[2].stats.starttime: continue
        if not st[0].stats.endtime   == st[1].stats.endtime   == st[2].stats.endtime:   continue
        if len(st[0])!=int(win_size*100+1): continue
        if st.max()[0]==0.0 or st.max()[1]==0.0 or st.max()[2]==0.0: continue
        # add preprocessed time window
        st = preprocess_stream(st)
        stream_list.append(st)
    return stream_list


def preprocess_stream(stream):
    """preprocess: rmean + rtr + normalize"""
    stream = stream.detrend('constant') # rmean + rtr in SAC
    return stream.normalize()


def repick(stream_data, tp0, ts0, repick_param):
    
    # get repick params
    pick_win  = repick_param['pick_win'] 
    search_dt = repick_param['search_dt']
    
    # calc tp
    if tp0 == -1: tp = -1
    elif int(100*tp0) <= search_dt +10: tp=tp0
    elif int(100*tp0) >= len(stream_data[2]) -10: tp=tp0
    else:
        z_idx   = int(100*tp0)
        z_range = range(z_idx-search_dt, z_idx+search_dt)
        tz = CF(stream_data[2], z_range, pick_win)
        if tz==-1: tp=-1
        else: tp = tp0 - 0.01 *search_dt + 0.01 *tz
    
    # calc ts
    if ts0 == -1: ts = -1
    elif int(100*ts0) <= search_dt +10: ts=ts0
    elif int(100*ts0) >= len(stream_data[0]) -10: ts=ts0
    else:
        xy_idx   = int(100*ts0)
        xy_range = range(xy_idx - search_dt,  xy_idx + search_dt) 
        tx = CF(stream_data[0], xy_range, pick_win)
        ty = CF(stream_data[1], xy_range, pick_win)
        if tx==-1 or ty==-1: ts=-1
        else: ts  = ts0 - 0.01 *search_dt + 0.01 *(tx + ty)/2
    
    return tp, ts

def CF(data, idx_range, pick_win):
    # use std as CF, then no filter needed
    snr = np.array([])
    for i in idx_range:
        if sum(data[max(0, i-pick_win):i])==0: return -1
        sta = np.std(data[i: min(len(data)-1, i+pick_win)])
        lta = np.std(data[max(0,i-pick_win): i])
        snri = sta /lta
        snr = np.append(snr, snri)
    return np.argmax(snr)


def run_det(win_gen, cfg, ckpt_dir):
    """
    run DetNet to output an event_list out of the generated windows
    """
    event_list = []
    prob_list  = []
    
    with tf.Session() as sess:
        # import DetNet
        inp_holder = {'data': tf.placeholder(tf.float32,
                            shape=(cfg.batch_size, 1, int(args.window_size*100+1), 3),
                            name='input_data')}
        DetNet = models.get('DetNet', inp_holder, cfg, ckpt_dir)
        DetNet.load(sess, None)
        
        n_events = 0
        time_start = time.time()
        for idx, win in enumerate(win_gen):
            # run DetNet
            to_fetch = [DetNet.layers['class_prediction'],
                        DetNet.layers['class_prob']]     
            
            feed_dict = {inp_holder['data']: fetch_window_data(win, 1, len(win[0])-1, 0)}
            [class_pred, class_prob] = sess.run(to_fetch, feed_dict)
            
            is_event = class_pred[0] > 0
            if is_event:
                n_events += 1
                event_list.append(idx)
                prob_list.append(class_prob[0][1])
                print 'detected events: {} to {} ({}%)'.format(win[0].stats.starttime, 
                                                                win[0].stats.endtime,
                                                                round(class_prob[0][1]*100, 2))
            if idx % 1000 ==0:
                print "Analyzing {} records".format(win[0].stats.starttime)
    
    print "processed {} windows".format(idx)
    print "DetNet Run time: ", time.time() - time_start
    print "found {} events".format(n_events)
    
    tf.reset_default_graph()
    return event_list, prob_list


def run_ppk(win_gen, event_list, prob_list, cfg, ckpt_dir, output, step_param, repick_param):
    """
    run PpkNet to ppk the detected events
    """
    # get step_param
    win_len     = step_param['win_len']
    step_len    = step_param['step_len']
    step_stride = step_param['step_stride']
    num_step    = step_param['num_step']
    
    with tf.Session() as sess:
        # set up PpkNet model and validation metrics
        inp_holder = {'data': tf.placeholder(tf.float32,
                            shape=(cfg.batch_size, num_step, step_len+1, 3),
                            name='input_data')}
        PpkNet = models.get('PpkNet', inp_holder, cfg, ckpt_dir)
        PpkNet.load(sess, None)
        to_fetch = PpkNet.layers['class_prediction']
        time_start = time.time()
        n_events=0
        previews_pick=0
        for i, idx in enumerate(event_list):
            # pick the time windows with P lie in first half
            if previews_pick==idx-1\
                or (idx+1 in event_list\
                and prob_list[i] < prob_list[i+1]): 
                continue
            else:
                # run PpkNet
                to_fetch = PpkNet.layers['class_prediction']
                win = win_gen[idx]
                win_data = fetch_window_data(win, num_step, step_len, step_stride)
                win_t0 = win[0].stats.starttime
                feed_dict = {inp_holder['data']: win_data}
                class_pred = sess.run(to_fetch, feed_dict)[0]
                
                # decode to relative time (sec) to win_t0
                pred_p = np.where(class_pred==1)[0]
                pred_s = np.where(class_pred==2)[0]
                step_width   = 0.01 *step_len    # in sec
                stride_width = 0.01 *step_stride # in sec
                if len(pred_p)>0:
                    if pred_p[0]==0: tp0 = step_width/2
                    else:            tp0 = step_width + stride_width/2 + stride_width *(pred_p[0]-1)
                else: tp0 = -1; tp = -1
                if len(pred_s)>0:
                    ts0 = step_width + stride_width/2 + stride_width *(pred_s[0]-1)
                else: ts0 = -1; ts = -1
                
                # repick beside org pick time
                st_data = fetch_window_data(win, 1, win_len, 1)[0][0]
                tp, ts = repick(st_data, tp0, ts0, repick_param)
                if not tp0==-1:
                    tp0 = win_t0 + tp0
                    tp  = win_t0 + tp
                if not ts0==-1:
                    ts0 = win_t0 + ts0
                    ts  = win_t0 + ts
                
                print 'picked phase time: tp={}, ts={}'.format(tp, ts)
                output.write(unicode('{},{},{},{},{}\n'.format(win[0].stats.station, tp, ts, tp0, ts0)))
                n_events+=1
                previews_pick=idx
    
    print "Picked {} events".format(n_events)
    print "PpkNet Run time: ", time.time() - time_start
    tf.reset_default_graph()
    return


def main(args):
    
    win_len     = 30 *100     # in seconds
    step_len    = 100         # length of each time step (frame size)
    step_stride = step_len/2 # half overlap of time steps
    num_step    = -(step_len/step_stride-1) + win_len /step_stride
    
    step_param  = {'win_len':    win_len,
                  'step_len':    step_len,
                  'step_stride': step_stride,
                  'num_step':    num_step}
    repick_param = {'pick_win' : 100, # win_len, in points
                    'search_dt': 50}  # search range, in points
    
    # i/o files
    stream_files = sorted(glob.glob(args.stream_paths))
    if os.path.exists(args.out_file): 
        os.unlink(args.out_file)
    output = open(args.out_file, 'a')
    
    # process all streams
    for stream_file in stream_files:
        # Load stream
        print 'loading file {}'.format(stream_file)
        net, sta, time, chn = stream_file.split('.')
        
        # read three channel
        sts_xyz = sorted(glob.glob(net + '*%s*%s*'%(sta, time)))
        if len(sts_xyz)<3: print 'missing trace!'; continue
        stream = Stream(traces = [read(sts_xyz[0])[0],
                                  read(sts_xyz[1])[0], 
                                  read(sts_xyz[2])[0]])
        
        # slice into 30s time window
        print 'generating {}s time windows'.format(args.window_size)
        total_time_in_sec = 24 *60 *60 #TODO
        max_windows = 1+ (total_time_in_sec - args.window_size) / args.window_step
        start_time = UTCDateTime(time[0:7])
        win_gen  = win_slide(stream, start_time, args.window_size, args.window_step, max_windows)
        
        cfg = config.Config()
        cfg.batch_size = 1
        # run det
        event_list, prob_list = run_det(win_gen, cfg, args.det_ckpt)
        # run ppk
        run_ppk(win_gen, event_list, prob_list, cfg, args.ppk_ckpt, output, step_param, repick_param)
    
    output.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_paths", type=str,
                        default='/data/WC_mon78/*.*HZ',
                        help="paths to stream files")
    parser.add_argument("--det_ckpt", type=str,
                        default='/home/zhouyj/Documents/CONEDEP/output/basemodel/DetNet',
                        help="path to directory of checkpoints")
    parser.add_argument("--ppk_ckpt",type=str,
                        default='/home/zhouyj/Documents/CONEDEP/output/basemodel/PpkNet',
                        help="path to directory of checkpoints")
    parser.add_argument("--window_size",type=int,default=30.0,
                        help="size of the window to analyze")
    parser.add_argument("--window_step",type=int,default=15.0,
                        help="step between windows to analyze")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to analyze")
    parser.add_argument("--out_file",type=str,
                        default="/home/zhouyj/Desktop/AI_picker/run_AIpicker/output/phase_all.dat",
                        help="output phase file")
    args = parser.parse_args()
    main(args)
