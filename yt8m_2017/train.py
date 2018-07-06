from models import *
from utilities import *
from keras.optimizers import *

if __name__ == '__main__':
    # Trainig scheme: Adam(lr=1e-3) -> Adam(lr=1e-4) -> Adam(lr=5e-5) -> SGD(lr=1e-4, momentum=True)
    # I did it manually, my bad
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--model", type=int, default=7, help="model to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--opt", type=str, default='adam', help="model optimiser")
    args = parser.parse_args()

    models = [None, None, None, None, build_model_4, build_model_5, build_model_6, build_model_7]
    opts = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}
    opt = opts[args.opt](lr=args.lr)
    model_f = models[args.model]

    model = model_f(opt)
    train(model, args.model, load_weights=True)
