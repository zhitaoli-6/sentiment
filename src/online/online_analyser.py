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
    def __init__(self, path='tag_dist'):
        self.csf = CSF()
        self.psr = PSR()

        self.local_path = path

    def run(self):
        self.csf.train()
        while True:
            try:
                stats = self.psr.online_run(interval=10)
                if not stats:
                    continue
                tag2cnt = {'P':0, 'N':0, 'O':0}
                for txt in stats:
                    tag = self.csf.predict(txt)
                    tag2cnt[tag] += 1
                tag2dist =  {tag:cnt * 1.0 / len(stats) for tag,cnt in tag2cnt.items()}
                linfo('round online sentiment distribution: %s' % tag2dist)
                save(self.local_path, 'a', '%s\t%s\t%s\n' % (ET.format_time(time.localtime()), json.dumps(tag2dist), len(stats)))
            except Exception as e:
                lexcept('Unknown exception %s' % e)
            

def main():
    obj = SentimentAnalyser()
    obj.run()
    

if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/online_analyser.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise online analyser')
    main()
    logging.info('end')
