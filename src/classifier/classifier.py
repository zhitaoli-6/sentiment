#encoding=utf-8 
#!/usr/bin/env python 
import logging, sys
from bayes_classifier import BayesClassifier as ByC
from linear_classifier import LinearClassifier as LnC

sys.path.append('/home/lizhitao/repos/sentiment/src')
from stats_tool import StatsTool as ST

linfo = logging.info

def predict_rule_based(txt):
    '''
    return value:
    None: unknown
    P, N, O
    '''
    if '【' in txt and '】' in txt:
        return 'O'
    if '天气：' in txt and '空气质量：' in txt and '分享自' in txt:
        return 'O'
    if '幸运数字：' in txt and '综合运势：' in txt and '速配星座：' in txt and '分享自' in txt and '查看更多' in txt:
        return 'O'
    return None

class Classifier(object):
    def __init__(self, name, rule_enable=True):
        linfo('--------init classifier begin------------')
        self.name = name
        self.classifier = self.build_classifier(name)
        self.rule_enabled = rule_enable
        linfo('--------init classifier end------------')

    def predict(self, txts):
        if not isinstance(txts, list):
            raise Exception('Invalid parameter is given')
        tags = self.classifier.predict(txts)
        if len(tags) != len(txts):
            raise Exception('Different length of pred tags and given input txts')
        if self.rule_enabled:
            for i, txt in enumerate(txts):
                rule_tag = predict_rule_based(txt)
                if rule_tag:
                    tags[i] = rule_tag
        return tags
    
    def train(self):
        linfo('-----------%s begin train---------' % self)
        self.classifier.train()
        linfo('-----------%s end train--f---------' % self)

    def build_classifier(self, name):
        linfo('%s begin init' % self)
        obj = None
        if name == 'bayes':
            obj = ByC()
        elif name == 'lr':
            obj = LnC(name)
        elif name == 'svm':
            obj = LnC(name)
        if obj:
            linfo('%s init success' % self)
            return obj
        raise Exception('INVALID CLASSIFIER NAME')
    def __str__(self):
        return '[CLASSIFIER %s]' % self.name

if __name__ == '__main__':
    #txt = '阆中 今天(5月27日)天气：阵雨，16℃~18℃，微风≤3级，空气质量：轻度 (分享自@微心情) http://t.cn/zT7O7Ci'
    #txt =  txt.decode('utf-8').encode('gbk')
    txt = '今日(5月27日)综合运势：5，幸运颜色：蓝绿色，幸运数字：0，速配星座：魔羯座（分享自@微心情） 查看更多： http://t.cn/h5gw6'
    print txt
    print predict_rule_based(txt)
