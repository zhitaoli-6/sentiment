#encoding=utf-8 
#!/usr/bin/env python 
import logging
from bayes_classifier import BayesClassifier as ByC

linfo = logging.info

class Classifier(object):
    def __init__(self, classifier='bayes'):
        self.classifier = self.build_classifier(classifier)

    def predict(self, txt):
        return self.classifier.predict(txt)
    
    def train(self):
        self.classifier.train()

    def build_classifier(self, classifier_name):
        if classifier_name == 'bayes':
            return ByC()
        raise Exception('INVALID CLASSIFIER NAME')

        

