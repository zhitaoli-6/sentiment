#encoding=utf-8
import sys, json, logging
sys.path.insert(0, '/home/lizhitao/repos/sinaweibopy/')

from snspy import APIClient as APIC, SinaWeiboMixin as WbM
from const import APP_KEY, APP_SECRET, CALLBACK_URL, ACCESS_TOKEN, EXPIRES

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug

def create_client():
    client =  APIC(WbM, app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=CALLBACK_URL)
    client.set_access_token(ACCESS_TOKEN, EXPIRES)
    return client

def main():
    client = create_client()
    #paras = {'screen_name':'BrightestSirius', 'count':5, 'page':1}
    #rsp = client.statuses.user_timeline.get(screen_name=name)

    #rsp = client.statuses.home_timeline.get()
    paras = {'count':20}
    rsp = client.statuses.public_timeline.get(**paras)
    linfo('items count: %s' % len(rsp.statuses))
    for dic in rsp.statuses:
        linfo('user: %s. created at: %s. text: %s' % (dic.user['name'], dic['created_at'], dic['text']))

if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    linfo('-----------------')
    logging.info('begin supervise')
    #main()
    logging.exception(Exception("test"))
    logging.info('end')
