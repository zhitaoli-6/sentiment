#encoding=utf-8 
#!/usr/bin/env python 
import logging
from bayes_classifier import BayesClassifier as ByC
from linear_classifier import LinearClassifier as LnC

linfo = logging.info

class Classifier(object):
    def __init__(self, name):
        self.name = name
        self.classifier = self.build_classifier(name)

    def predict(self, txt):
        return self.classifier.predict(txt)
    
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
        if obj:
            linfo('%s init success' % self)
            return obj
        raise Exception('INVALID CLASSIFIER NAME')
    def __str__(self):
        return '[CLASSIFIER %s]' % self.name
