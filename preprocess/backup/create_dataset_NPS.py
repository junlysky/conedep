"""Creates tfrecords for NPS classification
in only one script
"""

import os, glob, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/Documents/CONEDEP/')

import numpy as np
import tensorflow as tf
from obspy.core import read
from obspy.core.stream import Stream
from convnet.data_pipeline import DataWriter


def main():

    window_size  = 1 #TODO
    stream_dir   = '/data/WC_AItrain/NPS_recog/frame1_all_norm/'
    out_dir0     = '/home/zhouyj/Documents/CONEDEP/data/train/NPS_frame1'
    stream_files = sorted(glob.glob(stream_dir + '*.SAC'))
    
    done_file = []
    for stream_file in stream_files:
        
        stream_file = os.path.split(stream_file)[-1]
        sta, time, label, n, chn, _ = stream_file.split('.')
        
        if   label=='0': out_class = 'Noise'
        elif label=='1': out_class = 'P_tail'
        elif label=='2': out_class = 'S_tail'
        out_dir = os.path.join(out_dir0, out_class)
        if not os.path.exists(out_dir):
           os.makedirs(out_dir)
        
        pha_id = '.'.join([label, time[0:7]]) # events happened on one day
        if pha_id not in done_file: done_file.append(pha_id)
        else: continue
        
        # Write event waveforms and labels in .tfrecords
        out_name = pha_id + ".tfrecords"
        out_path = os.path.join(out_dir, out_name)
        writer = DataWriter(out_path)
        
        # Load stream
        stx_paths = sorted(glob.glob(stream_dir + \
                           '*{}*.{}.*.BHE.SAC'.format(time[0:7], label)))
        sty_paths = sorted(glob.glob(stream_dir + \
                           '*{}*.{}.*.BHN.SAC'.format(time[0:7], label)))
        stz_paths = sorted(glob.glob(stream_dir + \
                           '*{}*.{}.*.BHZ.SAC'.format(time[0:7], label)))
        # check completeness
        if not len(stx_paths)==len(sty_paths)==len(stz_paths): 
            print('error in %s'%(pha_id)); continue
        
        for i in range(len(stx_paths)):
            stream = Stream(traces = [read(stx_paths[i])[0], 
                                      read(sty_paths[i])[0], 
                                      read(stz_paths[i])[0]])
            # check health
            if stream.max()[0]==0. or stream.max()[1]==0. or stream.max()[2]==0.:
                print 'brocken trace!'; continue
            n_traces  = len(stream)
            n_samples = len(stream[0].data)
            n_pts = stream[0].stats.sampling_rate * window_size + 1
            if (n_traces == 3) and (n_pts == n_samples):
                st_data = np.array([[stream[0].data, stream[1].data, stream[2].data]])
                # Write tfrecords
                writer.write(st_data, int(label))
                print "+ Creating tfrecords for phase {}, idx = {}".\
                        format(pha_id, i)
            else: print "Missing waveform for phase: %s" %(pha_id)
        
        writer.close()

if __name__ == "__main__": main()

