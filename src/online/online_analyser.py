#encoding=utf-8 
#!/usr/bin/env python 
import sys, os, json
import logging, time

from easy_tool import EasyTool as ET
sys.path.append('/home/lizhitao/repos/sentiment/src/classifier')

from classifier import Classifier as CSF
from public_stats_retriever import PublicStatRetriever as PSR

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug
lexcept = logging.exception

save = ET.write_file

class SentimentAnalyser(object):
    def __init__(self, names):
        linfo('begin init classfiers: %s' % names)
        self.psr = PSR()
        self.classifiers = [(CSF(name), 'tag_dist_%s' % name) for name in names]
        linfo('classifiers init succcessfully!')

    def run(self):
        for csf, path in self.classifiers:
            csf.train()
        while True:
            try:
                stats = self.psr.online_run(interval=10)
                if not stats:
                    continue
                stats = [x for x in  stats]
                for clf, path in self.classifiers:
                    tag2cnt = {'P':0, 'N':0, 'O':0}
                    pred_tags = clf.predict(stats)
                    for tag in pred_tags:
                        tag2cnt[tag] += 1
                    tag2dist =  {tag:cnt * 1.0 / len(stats) for tag,cnt in tag2cnt.items()}
                    linfo('%s-roundly online sentiment distribution: %s' % (clf, tag2dist))
                    save(path, 'a', '%s\t%s\t%s\n' % (ET.format_time(time.localtime()), json.dumps(tag2dist), len(stats)))
            except Exception as e:
                lexcept('Unknown exception %s' % e)

def main():
    obj = SentimentAnalyser(['bayes', 'lr'])
    obj.run()

if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/online_analyser.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise online analyser')
    main()
    logging.info('end')
