"""
Creates tfrecords dataset of CNN filtered time steps for PpkNet
This is done by running NPSNet on 30s time windows
"""

import os, glob, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/Documents/CONEDEP/')

import numpy as np
import tensorflow as tf
from obspy.core import read
from obspy.core.stream import Stream

from convnet.data_pipeline import DataWriter
import convnet.config as cfg
import convnet.models as models


def preprocess_stream(stream):
    stream = stream.detrend('constant')
    return stream.normalize()

def main():
    # hypo params
    win_size    = 30          # in seconds
    step_len    = 100         # length of each time step (frame size)
    step_stride = step_len/10 # half overlap of time steps
    num_step    = -(step_len/step_stride-1) + win_size*100 /step_stride
    
    stream_paths = '/data/WC_AItrain/Man_30s/All/Events/*Z.SAC'
    stream_dir   = '/data/WC_AItrain/Man_30s/All/Events/'
    output_dir   = '/home/zhouyj/Documents/CONEDEP/data/train/WC_man30s/all/ppk_raw_stride20'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stream_files = sorted(glob.glob(stream_paths))
    
    done_file = []
    for stream_file in stream_files:
        # one-day's data in a tfrecord file
        stream_file = os.path.split(stream_file)[-1]
        sta, time, chn, _ = stream_file.split('.')
        event_id = time[0:7] # events happened on one day
        if event_id not in done_file: done_file.append(event_id)
        else: continue
        
        # Write event waveforms and labels in .tfrecords
        output_name = 'event_' + 'man_' + event_id + ".tfrecords"
        output_path = os.path.join(output_dir, output_name)
        writer = DataWriter(output_path)
        
        # Load stream
        stx_paths = sorted(glob.glob(stream_dir + '*%s*BHE.SAC'%(event_id)))
        sty_paths = sorted(glob.glob(stream_dir + '*%s*BHN.SAC'%(event_id)))
        stz_paths = sorted(glob.glob(stream_dir + '*%s*BHZ.SAC'%(event_id)))
        if not len(stx_paths)==len(sty_paths)==len(stz_paths): 
            print('error in %s'%(event_id)); continue
        
        # for all streams:
        for i in range(len(stx_paths)):
            stream = Stream(traces = [read(stx_paths[i])[0], 
                                      read(sty_paths[i])[0], 
                                      read(stz_paths[i])[0]])
            stream = preprocess_stream(stream)
            # drop bad data
            if stream.max()[0]==0.0 or stream.max()[1]==0.0 or stream.max()[2]==0.0:
                print 'brocken trace!'; continue
            # stream info
            n_traces  = len(stream)
            n_samples = len(stream[0].data)
            n_pts = stream[0].stats.sampling_rate * win_size + 1
            lebel, p_time, s_time = 1, stream[0].stats.sac.t0, stream[0].stats.sac.t1 
            # convert to time_steps and write to TFRecord
            if (n_traces == 3) and (n_pts == n_samples):
                # def input of RNN
                input_array = np.zeros((num_step, n_traces, step_len+1), dtype=np.float32)
                # three chn data
                xdata = stream[0].data
                ydata = stream[1].data
                zdata = stream[2].data
                st_data = np.array([xdata, ydata, zdata])
                # filter all time steps by CNN
                for j in range(num_step):
                    idx_s = j * step_stride
                    idx_e = idx_s + step_len + 1
                    current_step = st_data[:, idx_s:idx_e]
                    input_array[j, :, :] = current_step
                
                # Write tfrecords
                writer.write(input_array, lebel, p_time, s_time)
                print("+ Creating tfrecords for event {}, idx = {}".format(event_id, i))
            else: print("Missing waveform for event: %s" %(event_id))
        writer.close()

if __name__ == "__main__": main()
