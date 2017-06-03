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

def build_mod_4():
    in1 = Input((128,), name='x1')
    x1 = fc_block(in1, 1024)
    x1 = fc_block(x1, 2048)

    in2 = Input((1024,), name='x2')
    x2 = fc_block(in2, 1024)
    x2 = fc_block(x2, 2048)

    x = merge([x1, x2], mode='concat', concat_axis=1)
    x = fc_block(x, 2048)
    x = fc_block(x, 4096)
    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    opt = optimizers.SGD(lr=1e-4, momentum=True)
    # opt = optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.summary()
    return model

def build_mod_5():
    in1 = Input((128,), name='x1')
    x1 = fc_block(in1, 2048)
    x1 = fc_block(x1, 2048)

    in2 = Input((1024,), name='x2')
    x2 = fc_block(in2, 2048)
    x2 = fc_block(x2, 2048)

    x = merge([x1, x2], mode='concat', concat_axis=1)
    x = fc_block(x, 4096)
    x = fc_block(x, 4096)
    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    # opt = optimizers.SGD(lr=1e-4, momentum=True)
    opt = optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.summary()
    return model

def build_mod_6():
    in1 = Input((128,), name='x1')
    x1 = fc_block(in1, 1024)
    x1 = fc_block(x1, 2048)

    in2 = Input((1024,), name='x2')
    x2 = fc_block(in2, 2048)
    x2 = fc_block(x2, 4096)

    x = merge([x1, x2], mode='concat', concat_axis=1)
    x = fc_block(x, 4096)
    x = fc_block(x, 4096)
    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    opt = optimizers.SGD(lr=1e-5, momentum=True)
    # opt = optimizers.Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.summary()
    return model

def build_mod_7():
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

def conv_pred(elid, el):
    out = []
    for i in range(len(elid)):
      t = 20
      idx = np.argsort(el[i])[::-1]
      out.append(elid[i] + ',' + ' '.join(['{} {:0.5f}'.format(j, el[i][j]) for j in idx[:t]]) + '\n')
    return out

def search_alpha():
    model_1 = build_mod_6()
    model_2 = build_mod_7()

    _, x1_val, x2_val, y_val = next(tf_itr('val', 20000))

    wfn = sorted(glob.glob('weights6/*.h5'))[-1]
    model_1.load_weights(wfn)
    print('loaded weight file: %s' % wfn)
    wfn = sorted(glob.glob('weights7/*.h5'))[-1]
    model_2.load_weights(wfn)
    print('loaded weight file: %s' % wfn)

    ypd_1 = model_1.predict({'x1': x1_val, 'x2': x2_val}, verbose=False, batch_size=100)
    ypd_2 = model_2.predict({'x1': x1_val, 'x2': x2_val}, verbose=False, batch_size=100)
    
    best_alpha = 0
    best_gap = 0
    for alpha in np.arange(0, 1.01, 0.01):
        ypd = alpha * ypd_1 + (1 - alpha) * ypd_2
        g = gap(ypd, y_val)
        if g > best_gap:
            best_gap = g
            best_alpha = alpha
        print('val GAP %0.5f; alpha: %f' % (g, alpha))
    print('val best GAP %0.5f; best_alpha: %f' % (best_gap, best_alpha))
    return best_alpha

def search_alpha_iter(ypd_1, ypd_2, y_val):
    best_alpha = 0
    best_gap = 0
    for alpha in np.arange(0, 1.01, 0.01):
        ypd = alpha * ypd_1 + (1 - alpha) * ypd_2
        g = gap(ypd, y_val)
        if g > best_gap:
            best_gap = g
            best_alpha = alpha
        print('val GAP %0.5f; alpha: %f' % (g, alpha))
    print('val best GAP %0.5f; best_alpha: %f' % (best_gap, best_alpha))
    return best_alpha

def search_alphas():
    _, x1_val, x2_val, y_val = next(tf_itr('val', 20000))

    models = [build_mod_4(), build_mod_5(), build_mod_6(), build_mod_7()]
    n_models = len(models)
    wfiles = [get_wfile(i) for i in [4, 5, 6, 7]]
    alphas = np.ones(4)

    for i in range(n_models):
        models[i].load_weights(wfiles[i])

    models_preds = [model.predict({'x1': x1_val, 'x2': x2_val}, verbose=0, batch_size=100) for model in models]
    
    ypd_1 = models_preds[0]
    ypd_2 = models_preds[1]
    alpha = search_alpha_iter(ypd_1, ypd_2, y_val)
    alphas[0] *= alpha
    alphas[1] *= (1 - alpha)
        
    ypd_1 = alpha * ypd_1 + (1 - alpha) * ypd_2
    ypd_2 = models_preds[2]
    alpha = search_alpha_iter(ypd_1, ypd_2, y_val)
    alphas[0] *= alpha
    alphas[1] *= alpha
    alphas[2] *= (1 - alpha)

    ypd_1 = alpha * ypd_1 + (1 - alpha) * ypd_2
    ypd_2 = models_preds[3]
    alpha = search_alpha_iter(ypd_1, ypd_2, y_val)
    alphas[0] *= alpha
    alphas[1] *= alpha
    alphas[2] *= alpha
    alphas[3] *= (1 - alpha)

    ypd = np.average(models_preds, weights=alphas, axis=0)
    print('val GAP {}; alphas: {}'.format(gap(ypd, y_val), alphas))

    return alphas

def get_wfile(i):
    return sorted(glob.glob('weights{}/*.h5'.format(i)))[-1]

def predict(out_file_location, alpha):
    with open(out_file_location, "w+") as out_file:
        model_1 = build_mod_6()
        model_2 = build_mod_7()

        wfn = sorted(glob.glob('weights6/*.h5'))[-1]
        model_1.load_weights(wfn)
        print('loaded weight file: %s' % wfn)
        wfn = sorted(glob.glob('weights7/*.h5'))[-1]
        model_2.load_weights(wfn)
        print('loaded weight file: %s' % wfn)
        cnt = 0

        out_file.write("VideoId,LabelConfidencePairs\n")
        for d in npz_itr('test'):
            print ('batch ', cnt)
            idx, x1_val, x2_val, _ = d
            ypd_1 = model_1.predict({'x1': x1_val, 'x2': x2_val}, verbose=1, batch_size=len(idx))
            ypd_2 = model_2.predict({'x1': x1_val, 'x2': x2_val}, verbose=1, batch_size=len(idx))
            ypd = alpha * ypd_1 + (1 - alpha) * ypd_2

            out = conv_pred(idx, ypd)
            for line in out:
                out_file.write(line)
            out_file.flush()

            cnt += 1

def predict_with_alphas(out_file_location, alphas):
    with open(out_file_location, "w+") as out_file:
        models = [build_mod_4(), build_mod_5(), build_mod_6(), build_mod_7()]
        n_models = len(models)
        wfiles = [get_wfile(i) for i in [4, 5, 6, 7]]

        for i in range(n_models):
            models[i].load_weights(wfiles[i])

        cnt = 0

        out_file.write("VideoId,LabelConfidencePairs\n")
        for d in npz_itr('test'):
            if cnt % 500 == 0:
                print ('batch ', cnt)
            idx, x1_val, x2_val, _ = d

            models_preds = [model.predict({'x1': x1_val, 'x2': x2_val}, verbose=0, batch_size=len(idx)) for model in models]
            preds = np.average(models_preds, weights=alphas, axis=0)
            assert preds.shape[0] == len(idx)

            out = conv_pred(idx, preds)
            for line in out:
                out_file.write(line)
            out_file.flush()

            cnt += 1

def predict_mean_all_models(out_file_location):
    with open(out_file_location, "w+") as out_file:
        models = [build_mod_4(), build_mod_5(), build_mod_6(), build_mod_7()]
        n_models = len(models)
        wfiles = [get_wfile(i) for i in [4, 5, 6, 7]]

        for i in range(n_models):
            models[i].load_weights(wfiles[i])

        cnt = 0

        out_file.write("VideoId,LabelConfidencePairs\n")
        for d in npz_itr('test'):
            if cnt % 500 == 0:
                print ('batch ', cnt)
            idx, x1_val, x2_val, _ = d

            models_preds = [model.predict({'x1': x1_val, 'x2': x2_val}, verbose=0, batch_size=len(idx)) for model in models]
            preds = np.mean(models_preds, axis=0)
            
            # print (np.array(models_preds).shape)
            assert preds.shape[0] == len(idx)

            out = conv_pred(idx, preds)
            for line in out:
                out_file.write(line)
            out_file.flush()

            cnt += 1

if __name__ == '__main__':
    # alpha = search_alpha()
    # predict('../output/keras_merge_[67]_alpha[{}].csv'.format(alpha), alpha)
    # predict_mean_all_models('../output/keras_mean_[4567].csv')
    # search_alphas()
    predict_with_alphas('../output/keras_weighted_mean_[4567].csv', alphas=[0.141504, 0.287296, 0.2412, 0.33])
