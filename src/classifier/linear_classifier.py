#encoding=utf-8 
#!/usr/bin/env python 
import sys, os, json
import logging, time, random, math

sys.path.append('/home/lizhitao/repos/sentiment/src')
from const import project_dir, INDEX2TAG, linear_model_config as config
from easy_tool import EasyTool as ET
from stats_tool import StatsTool as ST
from liblinear_helper import LinearModelInputHelper as LMH
from metric import cal_metric

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug

model_dir = '%s/src/model/' % project_dir
model_prefix = '%slinear_model_' % model_dir

test_data_path = '%slinear_classifier_test_data' % model_dir
predict_data_path = '%slinear_classifier_predict_data' % model_dir


class LinearClassifier(object):
    def __init__(self, name):
        #linfo('begin init LinearClassifier: %s' % name) 
        if name not in ['lr', 'svm']:
            raise Exception('NOT IMPLEMENTED LINEAR CLASSIFIER')
        self.model_path = '%s%s' % (model_prefix, name)
        self.model_helper = LMH()
        self.test_tmp_path = '%s_%s' % (test_data_path, name)
        self.predict_tmp_path = '%s_%s' % (predict_data_path, name)

    def predict(self, stats):
        if isinstance(stats, str):
            stats = [stats]
        if not isinstance(stats, list):
            raise Exception('INVALID Parameter is given. %s' % stats)
        if not stats:
            return None
        ET.write_file(self.test_tmp_path, 'w', '')
        for txt in stats:
            features = self.model_helper.get_sparse_feature(txt)
            ET.write_file(self.test_tmp_path, 'a', '-1 %s\n' % ' '.join(features))
        cmd = '%s/linear_predict %s %s %s 1>>std.log 2>>err.log' % (model_dir, self.test_tmp_path, self.model_path, self.predict_tmp_path)
        linfo('predict cmd: %s' % cmd)
        ret = os.system(cmd)
        linfo('predict finish. return value: %s' % ret)
        if ret != 0:
            raise Exception('Fatal Error-Classifier predict FAIL')
        if os.path.exists(self.predict_tmp_path):
            with open(self.predict_tmp_path, 'r') as f:
                pred_tags = [line.strip() for line in f]
            ldebug('read predict results cnt: %s' % len(pred_tags))
            if len(pred_tags) != len(stats):
                raise Exception('Invalid pred results')
            os.system('rm %s' % self.predict_tmp_path)
            try:
                return map(lambda x: INDEX2TAG[int(x)], pred_tags)
            except:
                raise Exception('Invalid pred results')
        return None

    def train(self):
        if not os.path.exists(self.model_path):
            raise Exception('linear model path does not exist!!')
        linfo('linear model helper config(sparse model): %s' % config)
        self.model_helper.train_discret_model(**config)
        ldebug('linear models are trained manually now')


def main():
    test_path = '%s/test_data/tri_test_data' % project_dir
    stats = []
    with open(test_path, 'r') as f:
        for line in f:
            dic = json.loads(line.strip())
            stats.append(dic.items()[0])
    linfo('read test data cnt: %s' % len(stats))
    lr_model = LinearClassifier('lr')
    lr_model.train()
    txts = map(lambda x:x[1], stats)
    tags = map(lambda x:x[0], stats)
    pred_tags = lr_model.predict(txts)
    #cal_metric(tags, pred_tags)
    
if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise classifier')
    main()
    logging.info('end')


