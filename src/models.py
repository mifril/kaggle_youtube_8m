from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, merge
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers


# based on https://www.kaggle.com/drn01z3/keras-baseline-on-video-features-0-7941-lb
def fc_block(x, n=1024, d=0.2):
    x = Dense(n, kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x

def build_model_4(opt=None):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 1024)
    l_rgb = fc_block(l_rgb, 2048)

    l_merge = merge([l_audio, l_rgb], mode='concat', concat_axis=1)
    l_merge = fc_block(l_merge, 2048)
    l_merge = fc_block(l_merge, 4096)
    out = Dense(4716, activation='sigmoid', name='output')(l_merge)

    model = Model(input=[in_audio, in_rgb], output=out)
    if opt == None:
        opt = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.summary()
    return model

def build_model_5(opt=None):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 2048)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_rgb = fc_block(l_rgb, 2048)

    l_merge = merge([l_audio, l_rgb], mode='concat', concat_axis=1)
    l_merge = fc_block(l_merge, 4096)
    l_merge = fc_block(l_merge, 4096)
    out = Dense(4716, activation='sigmoid', name='output')(l_merge)

    model = Model(input=[in_audio, in_rgb], output=out)
    if opt == None:
        opt = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.summary()
    return model

def build_model_6(opt=None):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_rgb = fc_block(l_rgb, 4096)

    l_merge = merge([l_audio, l_rgb], mode='concat', concat_axis=1)
    l_merge = fc_block(l_merge, 4096)
    l_merge = fc_block(l_merge, 4096)
    out = Dense(4716, activation='sigmoid', name='output')(l_merge)

    model = Model(input=[in_audio, in_rgb], output=out)
    if opt == None:
        opt = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.summary()
    return model

def build_model_7(opt=None):
    in_audio = Input((128,), name='audio')
    l_audio = fc_block(in_audio, 1024)
    l_audio = fc_block(l_audio, 2048)
    l_audio = fc_block(l_audio, 2048)

    in_rgb = Input((1024,), name='rgb')
    l_rgb = fc_block(in_rgb, 2048)
    l_audio = fc_block(l_audio, 2048)
    l_rgb = fc_block(l_rgb, 4096)

    l_merge = merge([l_audio, l_rgb], mode='concat', concat_axis=1)
    l_merge = fc_block(l_merge, 4096)
    l_merge = fc_block(l_merge, 4096)
    out = Dense(4716, activation='sigmoid', name='output')(l_merge)

    model = Model(input=[in_audio, in_rgb], output=out)
    if opt == None:
        opt = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.summary()
    return model
