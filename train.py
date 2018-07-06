import argparse
import gc

from tqdm import tqdm
import numpy as np

from dataset import Dataset
from models import get_model_f, OPTS, get_model, load_best_weights_max
from metric import gap


def train(args):
    train = Dataset(args, mode='train')
    val = Dataset(args, mode='val')

    model_f = get_model_f(args.model_name)
    opt = OPTS[args.opt](args.lr)
    model = get_model(model_f, opt)
    fold_wdir = load_best_weights_max(model, args.model_name, wdir=args.wdir)

    n_train = len(train)
    n_val = len(val)
    print('Train len:', n_train)
    print('Val len:', n_val)

    best_gap = 0

    for e in range(args.epochs):
        print('Epoch: ', e)

        train_generator = train.generator()
        data_iter = tqdm(enumerate(train_generator), total=int(np.ceil(n_train // args.batch_size)),
                         desc="Epoch {}".format(e), ncols=0)
        for i, data in data_iter:
            X, y = data
            loss = model.train_on_batch({'audio': X['audio'], 'rgb': X['rgb']}, {'output': y})
            data_iter.set_postfix(**{'loss': "{:.8f}".format(loss / (i + 1))})

        y_val = []
        y_pred = []
        for data in tqdm(val.generator()):
            X, y = data
            y_val.append(y)
            y_pred_batch = model.predict_on_batch({'audio': X['audio'], 'rgb': X['rgb']})
            y_pred.append(y_pred_batch)

        y_val = np.concatenate(y_val)
        y_pred = np.concatenate(y_pred)
        print(y_val.shape, y_pred.shape)
        cur_gap = gap(y_val, y_pred)
        print('val GAP {:.5}; epoch: {}'.format(cur_gap, e))

        if cur_gap > best_gap:
            fname = fold_wdir + '/{:.6f}-{:03d}.h5'.format(cur_gap, e)
            best_gap = cur_gap
            model.save_weights(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-m', '--model_name', type=str, default='dense4')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100000, help='Epochs number')
    parser.add_argument('-w', '--wdir', type=str, default=None, help='weights dir, if None - load by model_name')
    parser.add_argument('-o', '--opt', type=str, default='adam',
                        choices=['adam', 'rmsprop', 'sgd'])

    parser.add_argument('--use_tf_records', action='store_true')

    parser.add_argument('-p', '--patience', type=int, default=5, help='LRSheduler patience')
    parser.add_argument('-es', '--early_stopping', type=int, default=12, help='Early Stopping patience')
    parser.add_argument('-r', '--lr_reduce_rate', type=int, default=0.5, help='LRSheduler reduce rate')
    args = parser.parse_args()

    args.model_name = 'dense4'
    args.use_tf_records = True

    train(args)
    gc.collect()
