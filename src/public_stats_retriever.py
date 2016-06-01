#encoding=utf-8
#!/usr/bin/env python
import sys, json, logging, time

sys.path.insert(0, '/home/lizhitao/repos/sinaweibopy/')
from snspy import APIClient as APIC, SinaWeiboMixin as WbM
from const import APP_KEY, APP_SECRET, CALLBACK_URL, ACCESS_TOKEN, EXPIRES
from easy_tool import EasyTool as ET
from stats_tool import StatsTool as ST

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
    '''
        Online: analyse
        Offline: download states data on local disk
    '''
    def __init__(self, path='UNKNOWN'):
        self.client = create_client()
        self.path = path
        self.total_cnt = 0

    def online_run(self, interval=10, peroid=0.5, quiet=True):
        '''
        return value: [(city, stat)...]
        '''
        stats_set = set()
        stats = []
        now = peroid
        cnt = 0
        while now < interval:
            try:
                rsp = self.retrieve('on', quiet=quiet)
                cnt += 1
                if rsp:
                    for dic in rsp:
                        if dic['id'] not in stats_set:
                            city = ST.parse_spatial(dic)
                            item = (city, dic['text'])
                            stats.append(item)
                            stats_set.add(dic['id'])
            except Exception as e:
                logging.exception(e)
            now += peroid
            time.sleep(peroid*60)
        linfo('online analysis %s new stats retrieved. retrieve cnt: %s' % (len(stats), cnt))
        return stats

    def offline_run(self, peroid=10):
        while True:
            try:
                self.retrieve('off')
            except Exception as e:
                logging.exception(e)
            logging.info('sleep for %smin...' % peroid)
            time.sleep(peroid*60)
        
    def retrieve(self, line, quiet=False):
        if line not in ['on', 'off']:
            raise Exception('INVALID PARAMETER IS GIVEN WHEN RETRIEVE STATES')
        if not quiet:
            linfo('-----------------')
            linfo('begin retrieve once')
        paras = {'count':200}
        rsp = self.client.statuses.public_timeline.get(**paras)
        self.total_cnt += len(rsp.statuses)
        if not quiet:
            linfo('retrieve items count: %s. total_cnt: %s' % (len(rsp.statuses), self.total_cnt))
        if line == 'on':
            return rsp.statuses
        for dic in rsp.statuses:
            write(self.path, 'a', '%s\n' % json.dumps(dic))
        return None

def main():
    #paras = {'screen_name':'BrightestSirius', 'count':5, 'page':1}
    #rsp = client.statuses.user_timeline.get(screen_name=name)
    #rsp = client.statuses.home_timeline.get()
    obj = PublicStatRetriever(path='../stats/test_public_stats')
    obj.offline_run(peroid=0.5)

if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment_public_states_retriever.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('begin supervise public states retriever')
    main()
    logging.info('end')
