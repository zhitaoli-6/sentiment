#encoding=utf-8
#!/usr/bin/env python
import sys, json, logging, time

sys.path.insert(0, '/home/lizhitao/repos/sinaweibopy/')
from snspy import APIClient as APIC, SinaWeiboMixin as WbM
from const import APP_KEY, APP_SECRET, CALLBACK_URL, ACCESS_TOKEN, EXPIRES
from tool import EasyTool as ET

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug
write = ET.write_file

def create_client():
    client =  APIC(WbM, app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=CALLBACK_URL)
    client.set_access_token(ACCESS_TOKEN, EXPIRES)
    return client

class PublicStatRetriever(object):
    def __init__(self, path):
        self.client = create_client()
        self.path = path
        self.total_cnt = 0
        
    def run(self, interval=10):
        while True:
            try:
                self.retrieve()
            except Exception as e:
                logging.exception(e)
            logging.info('sleep for %smin...' % interval)
            time.sleep(interval*60)
        
    def retrieve(self):
        linfo('-----------------')
        linfo('begin retrieve once')
        paras = {'count':200}
        rsp = self.client.statuses.public_timeline.get(**paras)
        self.total_cnt += len(rsp.statuses)
        linfo('retrieve items count: %s. total_cnt: %s' % (len(rsp.statuses), self.total_cnt))
        for dic in rsp.statuses:
            write(self.path, 'a', '%s\n' % json.dumps(dic))

def main():
    #paras = {'screen_name':'BrightestSirius', 'count':5, 'page':1}
    #rsp = client.statuses.user_timeline.get(screen_name=name)
    #rsp = client.statuses.home_timeline.get()
    obj = PublicStatRetriever('stats/public_stats')
    obj.run(interval=0.5)

if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment_public_states_retriever.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('begin supervise public states retriever')
    main()
    logging.info('end')
