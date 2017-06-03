from models import *
from utilities import *
from keras import optimizers

if __name__ == '__main__':
    # Adam(lr=1e-3) -> Adam(lr=1e-4) -> Adam(lr=5e-5) -> SGD(lr=1e-4, momentum=True)
    opt = optimizers.Adam(lr=1e-3)
    model = build_model_7(opt)
    train(model, 7, load_weights=True)
