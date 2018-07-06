import numpy as np
import tensorflow as tf
import os
import glob
from tqdm import tqdm

def tf2npz(tf_path, export_folder='C:\\data\\test'):
    vid_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []
    tf_basename = os.path.basename(tf_path)
    npz_basename = tf_basename[:-len('.tfrecord')] + '.npz'
    isTrain = '/test' not in tf_path

    for example in tf.python_io.tf_record_iterator(tf_path):      
        tf_example = tf.train.Example.FromString(example).features
        vid_ids.append(tf_example.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
        if isTrain:
            labels.append(np.array(tf_example.feature['labels'].int64_list.value))
        mean_rgb.append(np.array(tf_example.feature['mean_rgb'].float_list.value).astype(np.float16))
        mean_audio.append(np.array(tf_example.feature['mean_audio'].float_list.value).astype(np.float16))
        
    save_path = export_folder + '/' + npz_basename
    np.savez(save_path, 
             rgb=np.array(mean_rgb), 
             audio=np.array(mean_audio), 
             ids=np.array(vid_ids),
             labels=labels
            )

if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool() as p:
        p.map(tf2npz, glob.glob('../input/test/*.tfrecord'))
