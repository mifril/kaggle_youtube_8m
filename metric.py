import numpy as np
import keras.backend as K
from multiprocessing import Pool


# based on https://www.kaggle.com/drn01z3/keras-baseline-on-video-features-0-7941-lb
def ap_at_n(data):
    # based on https://github.com/google/youtube-8m/blob/master/average_precision_calculator.py
    y_true, y_pred = data
    n = 20
    total_num_positives = None

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("the shape of y_pred and y_true does not match.")

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be 'None' or a positive integer."
                             " It was '%s'." % n)

    ap = 0.0
    sortidx = np.argsort(y_pred)[::-1]

    if total_num_positives is None:
        numpos = np.size(np.where(y_true > 0))
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
        if y_true[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    return ap


def gap(y_true, y_pred):
    lst = zip(list(y_true), list(y_pred))

    with Pool() as pool:
        all_gap = pool.map(ap_at_n, lst)

    return np.mean(all_gap)
