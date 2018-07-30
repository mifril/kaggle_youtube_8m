import os
import glob

import keras.backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Reshape, Lambda
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


def dense1(dropout_val=0.3):
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


def dense2(dropout_val=0.3):
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


def dense3(dropout_val=0.3):
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


def dense4(dropout_val=0.3):
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


def dense5(dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 1024)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_rgb = fc_block(l_rgb, 4096)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 4096 * 2)
    l_merge = fc_block(l_merge, 2048)
    out = Dense(N_CLASSES, activation='sigmoid', name='output')(l_merge)

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model


def dense6(dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 512)
    l_audio = fc_block(in_audio, 1024)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_rgb = fc_block(l_rgb, 4096)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 4096 * 2)
    l_merge = fc_block(l_merge, 2048)
    out = Dense(N_CLASSES, activation='sigmoid', name='output')(l_merge)

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model

def dense7(dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_rgb = fc_block(l_rgb, 2048 * 2)
    l_rgb = fc_block(l_rgb, 2048 * 2)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 4096)
    l_merge = fc_block(l_merge, 4096 * 2)
    l_merge = fc_block(l_merge, 4096 * 2)
    out = Dense(N_CLASSES, activation='sigmoid', name='output')(l_merge)

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model


def expert1(in_audio, in_rgb):
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)

    l_rgb = fc_block(in_rgb, 1024)
    l_rgb = fc_block(l_rgb, 2048)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 2048)
    l_merge = fc_block(l_merge, 4096)
    return l_merge


def gate1(in_audio, in_rgb):
    return expert1(in_audio, in_rgb)


def moe_out(input, n_mixtures=3):
    gate_distribution = input[0]
    expert_distribution = input[1]

    # print(expert_distribution.shape, gate_distribution[..., :n_mixtures].shape)
    return K.sum(gate_distribution[..., :n_mixtures] * expert_distribution, axis=2)


def moe1(n_mixtures=3, dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    in_rgb = Input((1024,), name='rgb')
    expert = expert1(in_audio, in_rgb)
    expert_distribution = Dense(N_CLASSES * n_mixtures, activation='softmax', name='expert_activations')(expert)
    expert_distribution = Reshape((-1, n_mixtures))(expert_distribution)

    gate = gate1(in_audio, in_rgb)
    gate_distribution = Dense(N_CLASSES * (n_mixtures + 1), activation='sigmoid', name='gate_activations')(gate)
    gate_distribution = Reshape((-1, n_mixtures + 1))(gate_distribution)

    out = Lambda(moe_out, name='output')([gate_distribution, expert_distribution])
    # out = K.sum(gate_distribution[:, :n_mixtures] * expert_distribution, 1)

    # out = Reshape((-1, N_CLASSES))(out)

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model


def expert2(in_audio, in_rgb):
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)
    l_audio = fc_block(l_audio, 2048)

    l_rgb = fc_block(in_rgb, 2048)
    l_audio = fc_block(l_audio, 2048)
    l_rgb = fc_block(l_rgb, 4096)

    l_merge = concatenate([l_audio, l_rgb], axis=1)
    l_merge = fc_block(l_merge, 4096)
    l_merge = fc_block(l_merge, 4096)
    return l_merge


def moe2(n_mixtures=3, dropout_val=0.3):
    in_audio = Input((128,), name='audio')
    in_rgb = Input((1024,), name='rgb')
    expert = expert2(in_audio, in_rgb)

    expert_distribution = Dense(N_CLASSES * n_mixtures, activation='softmax', name='expert_activations')(expert)
    expert_distribution = Reshape((-1, n_mixtures))(expert_distribution)

    gate = concatenate([in_audio, in_rgb], axis=1)
    gate_distribution = Dense(N_CLASSES * (n_mixtures + 1), activation='sigmoid', name='gate_activations')(gate)
    gate_distribution = Reshape((-1, n_mixtures + 1))(gate_distribution)

    out = Lambda(moe_out, name='output')([gate_distribution, expert_distribution])

    model = Model(inputs=[in_audio, in_rgb], outputs=out)
    return model


def get_model_f(model_name):
    fmodels = {'dense1': dense1, 'dense2': dense2, 'dense3': dense3, 'dense4': dense4, 'dense5': dense5, 'dense6': dense6,
               'dense7': dense7, 'moe1': moe1, 'moe2': moe2}
    return fmodels[model_name]


def get_model(model_f, opt='adam'):
    model = model_f(dropout_val=0.3)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model
