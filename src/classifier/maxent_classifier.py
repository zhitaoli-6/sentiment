#encoding=utf-8 
#!/usr/bin/env python 
import sys, os
import logging, time, random, math

from nltk import MaxentClassifier, classify

from easy_tool import EasyTool as ET
from stats_tool import StatsTool as ST

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug


class MaxentClassifierHelper(object):
    def __init__(self, path):
        self._path = path
        self._feature_extract_config = ['unigram']

    def train(self,emoticon=True, cross_validation=False, fold_sz=10, test_path='../test_data/test_data'):
        self._train_xs, self._train_ys = ST.load_data(self._path)
        ST.replace_url(self._train_xs, fill=True)
        if not emoticon:
            ST.remove_emoticon(self._train_xs)
        self.gram2gid = self._discretize_gram2gid()

        if cross_validation:
            linfo('begin to cross train')
            self._cross_train(fold_sz)
        else:
            self._test_xs, self._test_ys = ST.load_data(test_path)
            ST.replace_url(self._test_xs, fill=True)

            test_set = [(self._feature_encoding(txt), tag) for txt, tag in zip(self._test_xs, self._test_ys)]

            classifier = self._train(self._train_xs, self._train_ys)
            linfo('maxent classifier precision: %.4f' % classify.accuracy(classifier, test_set))
    
    def _cross_train(self, fold_sz):
        rid2shard = ST.random_shardlize(fold_sz, len(self._train_xs), load=True)
        precision = 0
        for fid,sd in rid2shard.items():
            tmp_train_xs = [self._train_xs[i] for i in sd]
            tmp_train_ys = [self._train_ys[i] for i in sd]
            test_set = [(self._feature_encoding(self._train_xs[i]), self._train_ys[i]) for i in sd]
            classifier = self._train(tmp_train_xs, tmp_train_ys)
            p = classify.accuracy(classifier, test_set)
            linfo('maxent classifier precision: %.4f' % p)
            precision += p
        linfo('average maxent classifier precision: %.4f' % precision/fold_sz)

    def _train(self, txs, tys):
        #rid2shard = ST.random_shardlize(10, len(self._train_xs))
        train_set = [(self._feature_encoding(txt), tag) for txt, tag in zip(txs, tys)]
        return MaxentClassifier.train(train_set, algorithm='iis', max_iter=4)

    def _feature_encoding(self, txt):
        bags = ST.retrieve_feature(txt, feature_extract_config=self._feature_extract_config)
        #fs = {x:0 for x in gram2gid}
        fs = {}
        for gram in bags:
            if gram in self.gram2gid:
                fs[self.gram2gid[gram]] = 1
        return fs
    
    def _discretize_gram2gid(self):
        w2id = {} 
        for txt in self._train_xs:
            bags = ST.retrieve_feature(txt, feature_extract_config=self._feature_extract_config)
            for w in bags:
                if w not in w2id:
                    w2id[w] = len(w2id) + 1
        linfo('grams cnt: %s' % len(w2id))
        return w2id




def main():
    obj = MaxentClassifierHelper('../train_data/train_data')
    obj.train(cross_validation=True, emoticon=False)
    
if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise classifier')
    main()
    logging.info('end')
