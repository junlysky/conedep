#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : create_dataset_noise.py
# Creation Date : 05-12-2016
# Last Modified : 08-09-2017 by Yijian Zhou
# Author: Thibaut Perol <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""Creates tfrecords dataset of events trace and their cluster_ids.
i
This is done by loading a dir of .mseed and one catalog with the
time stamps of the events and their cluster_id
Note: modify this code to accomplish your own task
"""

import os, glob, sys
sys.path.append('/home/zhouyj/Documents/CONEDEP/')
import numpy as np
from convnet.data_pipeline import DataWriter
import tensorflow as tf
from obspy.core import read
from obspy.core.stream import Stream

def preprocess_stream(stream):
    stream = stream.detrend('constant')
    return stream.normalize()

def main():

    window_size = 30
    stream_dir = '/data/WC_AItrain/Man_30s/Noise_aug/'
    output_dir = '/home/zhouyj/Documents/CONEDEP/data/train/WC_man30s/negative_aug/'
    stream_files = glob.glob(stream_dir + '*.SAC')

    done_file = []; num=0
    for stream_file in stream_files:

        stream_file = os.path.split(stream_file)[-1]
        # for files generated by 'cutevents'
#        net, sta, _1, chn, _2, event_id, _3 = stream_file.split('.')
        sta, time, chn, _ = stream_file.split('.')
        event_id = time[0:9]
        if event_id not in done_file: done_file.append(event_id)
        else: continue

        # Write event waveforms and labels in .tfrecords
        output_name = 'noise_' + 'man_' + event_id + ".tfrecords"
        output_path = os.path.join(output_dir, output_name)
        writer = DataWriter(output_path)

        # Load stream
        stx_paths = sorted(glob.glob(stream_dir + '*%s*.BHE.SAC'%(event_id)))
        sty_paths = sorted(glob.glob(stream_dir + '*%s*.BHN.SAC'%(event_id)))
        stz_paths = sorted(glob.glob(stream_dir + '*%s*.BHZ.SAC'%(event_id)))
        if not len(stx_paths)==len(sty_paths)==len(stz_paths): print('error in %s'%(event_id)); continue
        for i in range(len(stx_paths)):
          stream = Stream(traces = [read(stx_paths[i])[0], read(sty_paths[i])[0], read(stz_paths[i])[0]])
#          stream = preprocess_stream(stream)

          n_traces = len(stream)
          # If there is not trace skip this waveform
          if n_traces < 3: print 'missing trace!'; continue
          n_samples = len(stream[0].data)
          n_pts = stream[0].stats.sampling_rate * window_size + 1
          if (n_traces == 3) and (n_pts == n_samples):
            # Write tfrecords
            writer.write(stream, 0, 0.0, 0.0)
            num+=1; print("+ Creating tfrecords for noise {}, num = {}".format(event_id,num))
          else: print("Missing waveform for noise: %s" %(event_id))

        writer.close()

if __name__ == "__main__": main()

