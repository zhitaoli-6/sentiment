#encoding=utf-8 
#!/usr/bin/env python 
import sys, os
import json, logging, time, copy, random, math

from const import TAG2INDEX, project_dir, feature_config, linear_model_config as model_config
from easy_tool import EasyTool as ET
from stats_tool import StatsTool as ST
reload(sys)
sys.setdefaultencoding('utf-8')


linfo = logging.info
ldebug = logging.debug

#optimize: svm L2RL2L_Dual with unigram and bigram mixed. precision 74.8151

class LinearModelInputHelper(object):
    def __init__(self, ct='tri', prefix=''):
        if prefix and prefix != 'Dg_':
            raise Exception('INVALID PREFIX GIVEN!!!')
        self.flag_prefix = prefix
        self.train_data_path = '%s/train_data/%s%s_train_data' % (project_dir, prefix, ct)
        if ct not in ['bi', 'tri']:
            raise Exception('INVALID Classifier Type')
        self.classifier_type = ct
        self.tag2index = TAG2INDEX
        self._train_xs, self._train_ys = ST.load_data(self.train_data_path)
        self._train_ys = map(lambda x: self.tag2index[x], self._train_ys)

        #self._feature_extract_config = ['unigram', 'bigram']
        self._feature_extract_config = feature_config 
        linfo('feature extract config: %s' % self._feature_extract_config)
        linfo('classifier type %s' % ct)
        linfo('init %s success' % self)

    def train_discret_model(self, **config):
        linfo('begin train helper discret model: %s' % config)
        if not config['emoticon']:
            ST.remove_emoticon(self._train_xs)
        if not config['parenthesis']:
            ST.remove_parenthesis(self._train_xs)
        self.txt2bags = {}
        self.w2id = self.batch_extract_feature()
        linfo('end train helper discret model')

    def run(self,*args, **config):
        '''
        generate sparse train and test data
        '''
        try:
            self.train_discret_model(**config)
            self.format_train(**config)
            self.format_test(**config)
        except Exception as e:
            logging.exception(e)

    def format_test(self, emoticon=True, parenthesis=True):
        test_path='../test_data/%s_test_data' % self.classifier_type
        self._test_xs, self._test_ys = ST.load_data(test_path)
        linfo('begin preprocess test data, then sparse')
        self._raw_test_xs, self._test_xs = ST.preprocess(self._test_xs)
        #ST.replace_url(self._test_xs, fill='H')
        #ST.replace_target(self._test_xs, fill='T')
        self._test_ys = map(lambda x:self.tag2index[x], self._test_ys)
        self.format_sparse(self._test_xs, self._test_ys, '%s/test_data/%s%s_sparse_test_data_%s' % (project_dir, self.flag_prefix, self.classifier_type, 'icon' if emoticon else 'no_icon'))

    def format_train(self, emoticon=True, parenthesis=True):
        self.format_sparse(self._train_xs, self._train_ys, '%s/train_data/%s%s_sparse_train_data_%s' % (project_dir, self.flag_prefix, self.classifier_type, 'icon' if emoticon else 'no_icon'))

    def format_sparse(self,_xs, _ys, out_path):
        if os.path.exists(out_path):
            os.system('rm %s' % out_path)
        for txt, tag in zip(_xs, _ys):
            bags = self.get_feature(txt)
            features = self.discret_feature(bags)
            line = '%s %s' % (tag, ' '.join(features))
            ET.write_file(out_path, 'a', '%s\n' % line)

    def batch_extract_feature(self, word2cnt_path=None, cnt_threshold=10):
        self.w2id = {} 
        w2cnt = {}
        for txt in self._train_xs:
            bags = self.get_feature(txt, cache=True)
            for w in bags:
                w2cnt.setdefault(w, 0)
                w2cnt[w] += 1
                if w not in self.w2id:
                    self.w2id[w] = len(self.w2id) + 1
        if word2cnt_path:
            if os.path.exists(word2cnt_path):
                os.system('rm %s' % word2cnt_path)
            words = sorted(w2cnt.keys(), key=lambda x: w2cnt[x], reverse=True)
            for w in words:
                cnt = w2cnt[w]
                ET.write_file(word2cnt_path, 'a', '%s %s\n' % (w, cnt))
        #self.w2id = {w:d for w, d in self.w2id.items() if w2cnt[w] >= cnt_threshold}
        linfo('gram cnt: %s' % len(self.w2id))
        return self.w2id
    
    def get_feature(self, txt, cache=False):
        if txt in self.txt2bags:
            bags = self.txt2bags[txt]
        else:
            bags = ST.retrieve_feature(txt, feature_extract_config=self._feature_extract_config)
            if cache:
                self.txt2bags[txt] = bags
        return bags
    
    def discret_feature(self, bags):
        if not hasattr(self, 'w2id'):
            raise Exception('Not trained discret model')
        wids = [self.w2id[w] for w in bags if w in self.w2id]
        wids = set(wids)
        wids = sorted([x for x in wids])
        features = ['%s:%s' % (wid, 1) for wid in wids]
        return features

    def get_sparse_feature(self, txt):
        return  self.discret_feature(self.get_feature(txt))

    def __str__(self):
        return '[%s]' % LinearModelInputHelper.__name__

    def debug(self):
        ws_1 = set(self.batch_extract_feature().keys())
        ST.remove_emoticon(self._train_xs)
        ws_2 = set(self.batch_extract_feature().keys())
        linfo('uni_bi_icon_feature_cnt: %s. no_icon: %s' % (len(ws_1), len(ws_2)))
        rms =  ws_2 - ws_1
        for x in rms:
            print x
        #linfo(ws_1 - ws_2)

def main():
    #print dir(WorkClassifier)
    obj = LinearModelInputHelper('tri', 'Dg_')
    #obj.batch_extract_feature()
    #obj.debug()
    config = model_config
    obj.run(**config)

    #obj.train(emoticon=False, parenthesis=True)
    #obj.test(emoticon=False)
    
    
if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise liblinear')
    main()
    logging.info('end')

