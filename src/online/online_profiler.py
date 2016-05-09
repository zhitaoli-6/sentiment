#encoding=utf-8 
#!/usr/bin/env python 
import sys, os, json
import logging, time

from easy_tool import EasyTool as ET
sys.path.append('/home/lizhitao/repos/sentiment/src')
from public_stats_retriever import PublicStatRetriever as PSR

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug
lexcept = logging.exception

save = ET.write_file

stats_prefix = 'stats_tmp/'
class OnlineProfiler(object):
    def __init__(self):
        linfo('init %s' % self)
        self.psr = PSR()

    def select_stats(self, st_time, ed_time):
        while True:
            try:
                f_t = ET.format_time(time.localtime())
                if f_t > ed_time or f_t < st_time:
                    linfo('not valid time. sleep 600s')
                    time.sleep(600)
                    continue
                stats = self.psr.online_run(interval=1)
                if not stats:
                    continue
                f_t = ET.format_time(time.localtime())
                if f_t < ed_time and f_t > st_time:
                    tmp_path = '%stmp_%s' % (stats_prefix, f_t.replace(' ', '').replace('-', '').replace(':',''))
                    ET.write_file(tmp_path, 'a', '%s\n' % json.dumps(stats))
                    linfo('cache stats at %s' % f_t)
            except Exception as e:
                lexcept('Unknown exception %s' % e)
    
    def __str__(self):
        return OnlineProfiler.__name__

def main():
    obj = OnlineProfiler()
    obj.select_stats('2016-05-03 14:30:00', '2016-05-03 15:30:00')

if __name__ == '__main__':
    log_name = 'debug.log'
    logging.basicConfig(filename='/home/lizhitao/log/%s' % log_name,format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise')
    main()
    logging.info('end')
