import numpy as np
import numpy as np
import glob
import os

from metric import gap
import tensorflow as tf

FOLDER_TF = '../input/' #path to: train, val, test folders with *.tfrecord files
FOLDER_NPZ = 'C:\\data\\' #path to: train, val, test folders with *.npz files

def get_wfile(i):
    return sorted(glob.glob('weights{}/*.h5'.format(i)))[-1]

# based on https://www.kaggle.com/drn01z3/keras-baseline-on-video-features-0-7941-lb
def tf_iterator(tp='test', batch=1024):
    tfiles = sorted(glob.glob(os.path.join(FOLDER_TF, tp, '*tfrecord')))
    print('total files in {} {}'.format(tp, len(tfiles)))
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

def npz_iterator(tp='test'):
    files = sorted(glob.glob(os.path.join(FOLDER_NPZ, tp, '*npz')))
    print('total files in {} {}'.format(tp, len(files)))
    for fn in files:
        loaded = np.load(fn)
        labels = loaded['labels']
        labels_ohe = np.zeros((labels.shape[0], 4716)).astype(np.uint8)
        for i in range(labels.shape[0]):
            labels_ohe[i, labels[i]] = 1
        yield loaded['ids'], loaded['audio'].astype(np.float32), loaded['rgb'].astype(np.float32), labels_ohe

def train(model, wdir_id, load_weights=False, n_itr=500, n_epochs=100, n_val=20000):
    if not os.path.exists('weights' + str(wdir_id)):
        os.mkdir('weights' +  str(wdir_id))

    _, audio_val, rgb_val, y_val = next(tf_iterator('val', n_val))

    if load_weights:
        wfn = sorted(glob.glob('weights7/*.h5'))[-1]
        model.load_weights(wfn)
        print('loaded weight file: %s' % wfn)
  
    for e in range(n_epochs):
        print('Epoch: ', e)
        batch_count = 0
        for d in npz_iterator('train'):
            _, audio_train, rgb_train, y_train = d
            model.train_on_batch({'audio': audio_train, 'rgb': rgb_train}, {'output': y_train})
            batch_count += 1
            if batch_count % n_itr == 0:
                print('batch ', batch_count)
                y_prd = model.predict({'audio': audio_val, 'rgb': rgb_val}, verbose=False, batch_size=100)
                g = gap(y_prd, y_val)
                print('val GAP {:.5}; epoch: {}; batch: {}'.format(g, e, batch_count))
                model.save_weights('weights7/{:.5}_{}_{}.h5'.format(g, e, batch_count))

def conv_pred(elid, el):
    out = []
    for i in range(len(elid)):
      t = 20
      idx = np.argsort(el[i])[::-1]
      out.append(elid[i] + ',' + ' '.join(['{} {:0.5f}'.format(j, el[i][j]) for j in idx[:t]]) + '\n')
    return out

def search_alpha_itr(preds_1, preds_2, y_val):
    best_alpha = 0
    best_gap = 0
    for alpha in np.arange(0, 1.01, 0.01):
        preds = alpha * preds_1 + (1 - alpha) * preds_2
        g = gap(preds, y_val)
        if g > best_gap:
            best_gap = g
            best_alpha = alpha
        print('val GAP {:.5}; alpha: {:.5}'.format(g, alpha))
    print('val best GAP {:.5}; best_alpha: {:.5}'.format(best_gap, best_alpha))
    return best_alpha

def search_alpha(model_1, model_2, wdir_id1, wdir_id2, n_val=20000):
    _, audio_val, rgb_val, y_val = next(tf_iterator('val', n_val))

    model_1.load_weights(get_wfile(wdir_id1))
    model_2.load_weights(get_wfile(wdir_id2))

    preds_1 = model_1.predict({'audio': audio_val, 'rgb': rgb_val}, verbose=False, batch_size=100)
    preds_2 = model_2.predict({'audio': audio_val, 'rgb': rgb_val}, verbose=False, batch_size=100)
    
    return search_alpha_itr(preds_1, preds_2, y_val)

def search_alphas(models, wdir_ids, n_val=20000):
    _, audio_val, rgb_val, y_val = next(tf_iterator('val', n_val))

    wfiles = [get_wfile(i) for i in wdir_ids]
    alphas = np.ones(len(models))

    for i in range(len(models)):
        models[i].load_weights(wfiles[i])

    models_preds = [model.predict({'audio': audio_val, 'rgb': rgb_val}, verbose=0, batch_size=100)
                        for model in models]
    
    preds_1 = models_preds[0]
    for i in range(1, len(models)):
        preds_2 = models_preds[i]
        alpha = search_alpha_itr(preds_1, preds_2, y_val)
        for j in range(i):
            alphas[j] *= alpha
        alphas[i] *= (1 - alpha)

        preds_1 = alpha * preds_1 + (1 - alpha) * preds_2

    preds = np.average(models_preds, weights=alphas, axis=0)
    print('val GAP {}; alphas: {}'.format(gap(preds, y_val), alphas))

    return alphas

def predict_itr(out_file, models, alphas, n_itr=500):
    batch_count = 0
    with open(out_file_location, "w+") as out_file:
        out_file.write("VideoId,LabelConfidencePairs\n")
        for d in npz_iterator('test'):
            if batch_count % n_itr == 0:
                print ('batch ', batch_count)
            idx, audio_val, rgb_val, _ = d

            models_preds = [model.predict({'audio': audio_val, 'rgb': rgb_val}, verbose=0, batch_size=len(idx))
                                for model in models]
            preds = np.average(models_preds, weights=alphas, axis=0)
            assert preds.shape[0] == len(idx)

            out = conv_pred(idx, preds)
            for line in out:
                out_file.write(line)
            out_file.flush()

            batch_count += 1

def predict_wmean_2(model_1, model_2, wdir_id1, wdir_id2, out_file, alpha):
    model_1.load_weights(get_wfile(wdir_id1))
    model_2.load_weights(get_wfile(wdir_id2))
    predict_itr(out_file, [model_1, model_2], [alpha, 1 - alpha])
        
def predict_wmean(out_file, models, wdir_ids, alphas):
    wfiles = [get_wfile(i) for i in wdir_ids]
    for i in range(len(models)):
        models[i].load_weights(wfiles[i])
    predict_itr(out_file, models, alphas)

def predict_mean(models, wdir_ids, out_file):
    wfiles = [get_wfile(i) for i in wdir_ids]
    for i in range(len(models)):
        models[i].load_weights(wfiles[i])
    predict_itr(out_file, models, alphas=None)
