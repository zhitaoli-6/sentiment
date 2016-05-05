#encoding=utf-8 
#!/usr/bin/env python 
import logging
from bayes_classifier import BayesClassifier as ByC
from linear_classifier import LinearClassifier as LnC

linfo = logging.info




def predict_rule_based(txt):
    '''
    return value:
    None: unknown
    P, N, O
    '''
    if '【' in txt and '】' in txt:
        return 'O'
    return None

class Classifier(object):
    def __init__(self, name, rule_enable=True):
        self.name = name
        self.classifier = self.build_classifier(name)
        self.rule_enabled = rule_enable

    def predict(self, txts):
        if not isinstance(txts, list):
            raise Exception('Invalid parameter is given')
        tags = self.classifier.predict(txts)
        if not tags:
            return tags
        if len(tags) != len(txts):
            raise Exception('Different length of pred tags and given input txts')
        if self.rule_enabled:
            for i, txt in enumerate(txts):
                rule_tag = predict_rule_based(txt)
                if rule_tag:
                    tags[i] = rule_tag
        return tags
    
    def train(self):
        linfo('%s begin train' % self)
        self.classifier.train()
        linfo('%s end train' % self)

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
