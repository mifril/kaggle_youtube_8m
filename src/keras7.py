from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, merge
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
import numpy as np
import pandas as pd
import glob
import os

from time import time

import tensorflow as tf
from multiprocessing import Pool

FOLDER_TF = '../input/' #path to: train, val, test FOLDER_TFs with *tfrecord files
FOLDER_NPZ = 'C:\\data\\'

def ap_at_n(data):
    # based on https://github.com/google/youtube-8m/blob/master/average_precision_calculator.py
    predictions, actuals = data
    n = 20
    total_num_positives = None

    if len(predictions) != len(actuals):
        raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be 'None' or a positive integer."
                             " It was '%s'." % n)

    ap = 0.0

    sortidx = np.argsort(predictions)[::-1]

    if total_num_positives is None:
        numpos = np.size(np.where(actuals > 0))
    else:
        numpos = total_num_positives

    if numpos == 0:
        return 0

    if n is not None:
        numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
        r = min(r, n)
    for i in range(r):
        if actuals[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    return ap


def gap(pred, actual):
    lst = zip(list(pred), list(actual))
    # return ap_at_n(pred, actual)
    with Pool() as pool:
        all_gap = pool.map(ap_at_n, lst)

    return np.mean(all_gap)


def tf_itr(tp='test', batch=1024):
    tfiles = sorted(glob.glob(os.path.join(FOLDER_TF, tp, '*tfrecord')))
    print('total files in %s %d' % (tp, len(tfiles)))
    ids, aud, rgb, lbs = [], [], [], []
    for fn in tfiles:
        for example in tf.python_io.tf_record_iterator(fn):
            tf_example = tf.train.Example.FromString(example)

            ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
            rgb.append(np.array(tf_example.features.feature['mean_rgb'].float_list.value))
            aud.append(np.array(tf_example.features.feature['mean_audio'].float_list.value))

            yss = np.array(tf_example.features.feature['labels'].int64_list.value)
            out = np.zeros(4716).astype(np.int8)
            for y in yss:
                out[y] = 1
            lbs.append(out)
            if len(ids) >= batch:
                yield np.array(ids), np.array(aud), np.array(rgb), np.array(lbs)
                # yield np.array(rgb), np.array(lbs)
                ids, aud, rgb, lbs = [], [], [], []


def npz_itr(tp='test'):
    files = sorted(glob.glob(os.path.join(FOLDER_NPZ, tp, '*npz')))
    print('total files in %s %d' % (tp, len(files)))
    for fn in files:
        loaded = np.load(fn)
        labels = loaded['labels']
        labels_ohe = np.zeros((labels.shape[0], 4716)).astype(np.uint8)
        for i in range(labels.shape[0]):
            labels_ohe[i, labels[i]] = 1
        yield loaded['ids'], loaded['audio'].astype(np.float32), loaded['rgb'].astype(np.float32), labels_ohe

def fc_block(x, n=1024, d=0.2):
    x = Dense(n, init='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x

def build_mod():
    in1 = Input((128,), name='x1')
    x1 = fc_block(in1, 1024)
    x1 = fc_block(x1, 2048)
    x1 = fc_block(x1, 2048)

    in2 = Input((1024,), name='x2')
    x2 = fc_block(in2, 2048)
    x1 = fc_block(x1, 2048)
    x2 = fc_block(x2, 4096)

    x = merge([x1, x2], mode='concat', concat_axis=1)
    x = fc_block(x, 4096)
    x = fc_block(x, 4096)
    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    opt = optimizers.SGD(lr=1e-4, momentum=True)
    # opt = optimizers.Adam(lr=5e-5)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.summary()
    return model


def train():
    if not os.path.exists('weights7'): os.mkdir('weights7')

    batch = 1024
    n_itr = 100
    n_eph = 100

    _, x1_val, x2_val, y_val = next(tf_itr('val', 20000))

    model = build_mod()
    if True:
      wfn = sorted(glob.glob('weights7/*.h5'))[-1]
      model.load_weights(wfn)
      print('loaded weight file: %s' % wfn)
  
    for e in range(n_eph):
        print('Epoch: ', e)
        cnt = 0

        for d in npz_itr('train'):
            _, x1_trn, x2_trn, y_trn = d
            model.train_on_batch({'x1': x1_trn, 'x2': x2_trn}, {'output': y_trn})
            cnt += 1
            if cnt % n_itr == 0:
                print('batch ', cnt)
                y_prd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=False, batch_size=100)
                g = gap(y_prd, y_val)
                print('val GAP %0.5f; epoch: %d; iters: %d' % (g, e, cnt))
                model.save_weights('weights7/%0.5f_%d_%d.h5' % (g, e, cnt))

def conv_pred(elid, el):
    t = 20
    idx = np.argsort(el)[::-1]
    return elid + ',' + ''.join(['{} {:0.5f}'.format(i, el[i]) for i in idx[:t]])

def conv_pred2(elid, el):
    out = []
    for i in range(len(elid)):
      t = 20
      idx = np.argsort(el[i])[::-1]
      out.append(elid[i] + ',' + ' '.join(['{} {:0.5f}'.format(j, el[i][j]) for j in idx[:t]]) + '\n')
    return out

def predict(out_file_location):
    with open(out_file_location, "w+") as out_file:
      model = build_mod()

      wfn = sorted(glob.glob('weights7/*.h5'))[-1]
      model.load_weights(wfn)
      print('loaded weight file: %s' % wfn)
      cnt = 0

      out_file.write("VideoId,LabelConfidencePairs\n")
      for d in npz_itr('test'):
          print ('batch ', cnt)
          idx, x1_val, x2_val, _ = d
          ypd = model.predict({'x1': x1_val, 'x2': x2_val}, verbose=1, batch_size=len(idx))
          
          out = conv_pred2(idx, ypd)
          # with Pool() as pool:
              # out = pool.map(conv_pred, idx, ypd)
          for line in out:
              out_file.write(line)
          out_file.flush()
          
          cnt += 1

if __name__ == '__main__':
    # train()
    predict('../output/keras_nn_7.csv')
