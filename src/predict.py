from utilities import *
from models import *

if __name__ == '__main__':
    models = [build_model_4(), build_model_5(), build_model_6(), build_model_7()]
    wdir_ids = [4, 5, 6, 7]

    alphas = search_alphas(models, wdir_ids)
    predict_wmean('../output/keras_weighted_mean_[4567]_2.csv', models, wdir_ids, alphas)
