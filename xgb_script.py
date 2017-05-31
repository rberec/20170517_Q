import argparse
import functools
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser(description='XGB with Handcrafted Features')
    parser.add_argument('--save', type=str, default='XGB_leaky',
                        help='save_file_names')
    args = parser.parse_args()

    df_train = pd.read_csv('./input/train.csv.zip', compression='zip')

    X_train = pd.read_csv('./input/X_train.csv.gz', compression='gzip')
    y_train = df_train['is_duplicate'].values
    X_train = X_train.drop('is_duplicate', axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=4242)

    #UPDownSampling
    pos_train = X_train[y_train == 1]
    neg_train = X_train[y_train == 0]
    X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
    y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    print(np.mean(y_train))
    del pos_train, neg_train

    pos_valid = X_valid[y_valid == 1]
    neg_valid = X_valid[y_valid == 0]
    X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    print(np.mean(y_valid))
    del pos_valid, neg_valid


    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 7
    params['subsample'] = 0.6
    params['base_score'] = 0.2
    # params['scale_pos_weight'] = 0.2

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=50)
    # bst = xgb.train(params, d_train, 100, watchlist, early_stopping_rounds=50, verbose_eval=50)
    print(log_loss(y_valid, bst.predict(d_valid)))
    bst.save_model(args.save + '.mdl')


    df_test = pd.read_csv('./input/test.csv.zip', compression='zip')
    x_test = pd.read_csv('./input/X_test.csv.gz', compression='gzip')
    d_test = xgb.DMatrix(x_test)
    p_test = bst.predict(d_test)
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('outputX.csv', index=False)

if __name__ == '__main__':
    main()
