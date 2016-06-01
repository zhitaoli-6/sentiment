#encoding=utf-8 
#!/usr/bin/env python 
import sys, os, json
import logging, time

from easy_tool import EasyTool as ET
sys.path.append('/home/lizhitao/repos/sentiment/src/classifier')
sys.path.append('/home/lizhitao/repos/sentiment/src')

from stats_tool import StatsTool as ST
from classifier import Classifier as CSF
from public_stats_retriever import PublicStatRetriever as PSR

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug
lexcept = logging.exception

save = ET.write_file



sample_prefix = 'stats_realtime/'
profile_prefix = 'monitor/'
class SentimentAnalyser(object):
    def __init__(self, names):
        linfo('-----------begin init classfiers: %s--------------' % names)
        self.psr = PSR()
        self.classifiers = [(CSF(name), 'monitor/temporal/tag_dist_%s' % name) for name in names]

    def run(self, sample_enabled=False, profile_enabled=False):
        for csf, path in self.classifiers:
            csf.train()
        action_day = ET.format_time(time.localtime())[:10]
        action_total_cnt = 0
        if profile_enabled:
            self.reset_profile()
        while True:
            try:
                stats = self.psr.online_run(interval=10)
                if not stats:
                    continue
                
                linfo('-------roundly analysis----------')
                citys = map(lambda x:x[0], stats)
                stats = map(lambda x:x[1], stats)
                raw_stats, stats = ST.preprocess(stats)
                valid_ids = [i for i, txt in enumerate(stats) if txt]
                stats = map(lambda i:stats[i], valid_ids)
                raw_stats = map(lambda i:raw_stats[i], valid_ids)
                citys = map(lambda i:citys[i], valid_ids)
                f_t = ET.format_time(time.localtime())
                if sample_enabled:
                    sample_path = '%srealtime_%s' % (sample_prefix, f_t.replace(' ', '').replace('-', '').replace(':',''))
                    ET.write_file(sample_path, 'a', '%s\n' % json.dumps(raw_stats[:300]))
                
                #only one model supported at the same time now.
                for clf, path in self.classifiers:
                    tag2cnt = {'P':0, 'N':0, 'O':0}
                    pred_tags = clf.predict(stats)
                    for tag in pred_tags:
                        tag2cnt[tag] += 1
                    tag2dist =  {tag:cnt * 1.0 / len(stats) for tag,cnt in tag2cnt.items()}
                    linfo('%s-roundly online sentiment distribution: %s' % (clf, tag2dist))
                    f_time = ET.format_time(time.localtime())
                    today = f_time[:10]
                    action_total_cnt = (action_total_cnt + len(pred_tags)) if today == action_day else len(pred_tags)
                    save(path, 'a', '%s\t%s\t%s\n' % (f_time, json.dumps(tag2dist), len(stats)))
                    if profile_enabled:
                        self.update_profile_spatial(citys, pred_tags)
                        self.update_profile_topic(raw_stats, pred_tags)
                        if today != action_day:
                            self.save_profile(action_day)
                            self.reset_profile()
                            action_day = today
                    break
            except Exception as e:
                lexcept('Unknown exception %s' % e)

    def update_profile_topic(self, raw_stats, tags):
        for txt, tag in zip(raw_stats, tags):
            topic = ST.parse_topic(txt)
            if not topic:
                continue
            self.profile_topic.setdefault(topic, {"P":0,"N":0,"O":0})
            self.profile_topic[topic][tag] += 1
        
    def update_profile_spatial(self, citys, tags):
        for city, tag in zip(citys, tags):
            self.profile_city.setdefault(city, {"P":0,"N":0,"O":0})
            self.profile_city[city][tag] += 1

    def analyse_profile_outlier(self):
        pass
    
    def save_profile(self, action_day):
        #action_day format: 2016-01-01
        action_day = action_day.replace('-', '')
        city_path = '%scity/city_%s' % (profile_prefix, action_day)
        linfo('save spatial profile: %s' % city_path)
        ET.write_file(city_path, 'w', '')
        for city, dist in self.profile_city.items():
            ET.write_file(city_path, 'a', '%s,%s,%s,%s\n' % (city, dist['O'], dist['P'],dist['N']))
        
        topic_path = '%stopic/topic_%s' % (profile_prefix, action_day)
        linfo('save topic profile: %s' % topic_path)
        ET.write_file(topic_path, 'w', '')
        for tp, dist in self.profile_topic.items():
            ET.write_file(topic_path, 'a', '%s,%s,%s,%s\n' % (tp.replace(',', '-'), dist['O'], dist['P'],dist['N']))

    def reset_profile(self):
        '''
        '''
        #topic2tag_cnt
        self.profile_topic = {}
        #city2tag_cnt
        self.profile_city = {}

def main():
    obj = SentimentAnalyser(['lr'])
    obj.run(sample_enabled=True, profile_enabled=True)

if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/online_analyser.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise online analyser')
    main()
    logging.info('end')
