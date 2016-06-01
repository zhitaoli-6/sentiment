#encoding=utf-8 
#!/usr/bin/env python 
import sys, os
import logging, time, random, math

import numpy as np
from scipy import sparse

from sklearn.naive_bayes import BernoulliNB as BNB

sys.path.append('/home/lizhitao/repos/sentiment/src')
from easy_tool import EasyTool as ET
from stats_tool import StatsTool as ST
from metric import cal_metric
from const import TAG2INDEX as tag2index

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug

class BayesClassifier(object):

    def __init__(self, **config):
        linfo('config: %s' % config)
        self._path = config['train_path']
        self._feature_extract_config = config['feature']
        self._emoticon = config['emoticon']
        self.test_path='../../test_data/bi_test_data'
        self.clf = BNB(fit_prior=True)

    def train(self):
        self._train_xs, self._train_ys = ST.load_data(self._path)
        if not self._emoticon:
            ST.remove_emoticon(self._train_xs)
        self.gram2gid = self._discretize_gram2gid()
        X = self.build_sparse_X(self._train_xs)
        
        self.clf.fit(X, self._train_ys)
        
        self.real_test()
    
    def real_test(self):
        self._test_xs, self._test_ys = ST.load_data(self.test_path)
        ST.replace_url(self._test_xs, fill='H')
        ST.replace_target(self._test_xs, fill='T')
        #x_y = [(self.discret_txt(txt), y) for txt, y in zip(self._test_xs, self._test_ys)]
        test_mat = self.build_sparse_X(self._test_xs)
        self.accuracy(test_mat, self._test_ys)

    def predict_many(self, fs):
        return self.clf.predict(fs)

    def accuracy(self, test_mat, tags):
        #res = self.predict_many([fs for (fs, l) in pairs])
        #suc = [l == r for ((fs, l), r) in zip(pairs, res)]
        res = self.predict_many(test_mat)
        suc = [l == r for (l, r) in zip(tags, res)]
        linfo('precision: %.6f(%d/%d)' % (sum(suc) * 1.0/ len(res), sum(suc), len(res)))

        tags = map(lambda x:str(tag2index[x]), tags)
        res = map(lambda x:str(tag2index[x]), res)
        cal_metric(tags, res, False, False)
    
    def build_sparse_X(self, _xs):
        row_num = len(_xs)
        col_num = len(self.gram2gid)

        rows, cols = [], []
        total_cnt = 0
        for i,txt in enumerate(_xs):
            bags = ST.retrieve_feature(txt, feature_extract_config=self._feature_extract_config)
            for w in bags:
                if w in self.gram2gid:
                    wid = self.gram2gid[w]
                    rows.append(i)
                    cols.append(wid)
                    total_cnt += 1
        linfo('build scipy sparse matrice. total_valid_cnt: %s' % (total_cnt))
        row = np.array(rows)
        col = np.array(cols)
        data = np.array([1 for i in range(total_cnt)])
        mtx = sparse.csr_matrix((data, (row, col)), shape=(row_num, col_num))
        return mtx

    def discret_txt(self, txt):
        fs = [0 for x in range(len(self.gram2gid))]
        bags = ST.retrieve_feature(txt, feature_extract_config=self._feature_extract_config)
        for w in bags:
            if w in self.gram2gid:
                wid = self.gram2gid[w]
                fs[wid] = 1
        return fs

    def _discretize_gram2gid(self):
        w2id = {} 
        for txt in self._train_xs:
            bags = ST.retrieve_feature(txt, feature_extract_config=self._feature_extract_config)
            for w in bags:
                if w not in w2id:
                    w2id[w] = len(w2id) 
        linfo('grams cnt: %s' % len(w2id))
        return w2id



def main():
    config={'train_path':'../../train_data/Dg_bi_train_data', 'emoticon':False,'feature':['unigram']}
    obj = BayesClassifier(**config)
    obj.train()
    #obj.build_sparse_X()
    
if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise classifier')
    main()
    logging.info('end')
