import os
import glob

from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU

from metric import gap
from constants import MODELS_PATH, N_CLASSES

OPTS = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}


def load_best_weights_max(model, model_name, wdir=None, fold=None):
    if wdir is None:
        wdir = os.path.join(MODELS_PATH, str(model_name) + '/')
    if fold is not None:
        wdir = os.path.join(MODELS_PATH, str(model_name) + '/fold{}/'.format(fold))

    print('looking for weights in {}'.format(wdir))
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    else:
        files = sorted(glob.glob(os.path.join(wdir, '*.h5')))
        print(files)
        if len(files) > 0:
            wf = files[-1]
            model.load_weights(wf)
            print('loaded weights file: ', wf)
    return wdir


def fc_block(x, n=1024, d=0.2):
    x = Dense(n, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x


def dense1(opt, dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 1024)
    l_rgb = fc_block(l_rgb, 2048)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 2048)
    l_merge = fc_block(l_merge, 4096)
    out = Dense(N_CLASSES, activation='sigmoid', name='output')(l_merge)

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model


def dense2(opt, dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 2048)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_rgb = fc_block(l_rgb, 2048)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 4096)
    l_merge = fc_block(l_merge, 4096)
    out = Dense(N_CLASSES, activation='sigmoid', name='output')(l_merge)

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model


def dense3(opt, dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_rgb = fc_block(l_rgb, 4096)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 4096)
    l_merge = fc_block(l_merge, 4096)
    out = Dense(N_CLASSES, activation='sigmoid', name='output')(l_merge)

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model


def dense4(opt, dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_audio = fc_block(l_audio, 2048)
    l_rgb = fc_block(l_rgb, 4096)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 4096)
    l_merge = fc_block(l_merge, 4096)
    out = Dense(N_CLASSES, activation='sigmoid', name='output')(l_merge)

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model


def get_model_f(model_name):
    fmodels = {'dense1': dense1, 'dense2': dense2, 'dense3': dense3, 'dense4': dense4}
    return fmodels[model_name]


def get_model(model_f, opt='adam'):
    model = model_f(opt, dropout_val=0.3)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model
