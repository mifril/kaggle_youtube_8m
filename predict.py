import argparse
import gc
import os

from tqdm import tqdm
import numpy as np

from dataset import Dataset
from models import get_model_f, get_model, load_best_weights_max
from constants import OUTPUT_PATH


def process_preds(elid, el):
    out = []
    for i in range(len(elid)):
        t = 20
        idx = np.argsort(el[i])[::-1]
        out.append(elid[i] + ',' + ' '.join(['{} {:0.5f}'.format(j, el[i][j]) for j in idx[:t]]) + '\n')
    return out


def predict(args):
    test = Dataset(args, mode='test')
    if args.wmean:
        models = []
        raise NotImplementedError
    else:
        model_f = get_model_f(args.model_name)
        model = get_model(model_f)
        fold_wdir = load_best_weights_max(model, args.model_name, wdir=args.wdir)

    n_test = len(test)
    print('Test len:', n_test)

    out_path = os.path.join(OUTPUT_PATH, args.out_file)

    batch_count = 0
    with open(out_path, "w+") as f:
        f.write("VideoId,LabelConfidencePairs\n")
        for data in tqdm(test.generator(), total=int(np.ceil(n_test // args.batch_size))):
            idx, X, _ = data
            if args.wmean:
                raise NotImplementedError
                alphas = []
                models_preds = [model.predict({'audio': X['audio'], 'rgb': X['rgb']}, verbose=0, batch_size=args.batch_size)
                                for model in models]
                preds = np.average(models_preds, weights=alphas, axis=0)
            else:
                preds = model.predict({'audio': X['audio'], 'rgb': X['rgb']}, verbose=0, batch_size=args.batch_size)
            assert preds.shape[0] == len(idx)

            out = process_preds(idx, preds)
            for line in out:
                f.write(line)
            f.flush()

            batch_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='dense4')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('-w', '--wdir', type=str, default=None, help='weights dir, if None - load by model_name')
    parser.add_argument('--out_file', type=str, default='preds.csv')

    parser.add_argument('--wmean', action='store_true')

    parser.add_argument('--use_tf_records', action='store_true')
    args = parser.parse_args()

    args.model_name = 'dense4'
    args.use_tf_records = True

    predict(args)
    gc.collect()
