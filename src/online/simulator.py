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

stats_predict_detail_prefix = 'stats_simulate/'

class Simulator(object):
    def __init__(self, names):
        self.stats_dir = '/home/lizhitao/repos/sentiment/src/online/%s' % sys.argv[1]
        if not os.path.isdir(self.stats_dir):
            raise Exception('directory not exist')
        linfo('init simulator')
        self.stats = self.load_stats()
        linfo('begin init classfiers: %s' % names)
        self.psr = PSR()
        self.classifiers = [(CSF(name), 'simulator_tag_dist_%s' % name) for name in names]
        linfo('classifiers init succcessfully!')

    def run(self, detail=False):
        self.train()
        for name, stat in self.stats:
            try:
                for clf, path in self.classifiers:
                    linfo('----------roundly predict start-----------')
                    raw_stat, stat = ST.preprocess(stat)
                    union = [(raw,new) for raw, new in zip(raw_stat, stat) if new]
                    raw_stat = map(lambda x:x[0], union)
                    stat = map(lambda x:x[1], union)
                    pred_tags = clf.predict(stat)
                    if not pred_tags or len(pred_tags) != len(stat):
                        raise Exception('Predict Results Exception')
                    tag2dist = self.cal_tag_dist(pred_tags)
                    linfo('%s-roundly online sentiment distribution: %s' % (clf, tag2dist))
                    save(path, 'a', '%s\t%s\t%s\n' % (name, json.dumps(tag2dist), len(stat)))
                    if detail:
                        detail_path = '%s%s' % (stats_predict_detail_prefix, name)
                        if os.path.exists(detail_path):
                            os.system('rm %s' % detail_path)
                        for tag, txt in zip(pred_tags, raw_stat):
                            ET.write_file(detail_path, 'a', '%s -%s\n' % (tag, txt))
                            #print tag, '-%s' % txt
                    linfo('----------roundly predict end-----------')
            except Exception as e:
                lexcept('Unknown exception %s' % e)

    def debug_topic_public_stats(self, detail=False):
        #print '--------------------------------------'
        topic2txt = self.parse_topics_test_data()

        topics = sorted(topic2txt.keys(), key=lambda x: len(topic2txt[x]), reverse=True)
        #for tp in topics:
        #    print tp, len(topic2txt[tp])
        #for txt in topic2txt[u'#和颐酒店女生遇袭#']:
        #    print txt
        out_path = 'topic_test_tag_dist_format'
        self.train()
        for tp in topics:
            txts = topic2txt[tp]
            if len(txts) < 100:
                break
            txts = [x for x in txts]
            for csf, path in self.classifiers:
                #pred_tags = csf.predict(txts)
                #if not pred_tags or len(pred_tags) != len(txts):
                #    raise Exception('Predict Results Exception')
                #tag2dist = self.cal_tag_dist(pred_tags)

                txts_no_topic = map(lambda x:x.replace(tp, ''), txts)
                pred_tags = csf.predict(txts_no_topic)
                if not pred_tags or len(pred_tags) != len(txts):
                    raise Exception('Predict Results Exception')
                tag2dist = self.cal_tag_dist(pred_tags)
                ET.write_file(out_path, 'a', '%s,%s,%.4f,%.4f,%.4f\n' % (tp, len(txts), tag2dist['O'], tag2dist['P'], tag2dist['N']))
                if detail:
                    detail_path = 'topic_test_simulate'
                    for tag, txt in zip(pred_tags, txts):
                        ET.write_file(detail_path, 'a', '%s -%s\n' % (tag, txt))

    def debug_city_public_stats(self, detail=False):
        city2txt = self.parse_citys_test_data()
        return
        out_path = 'city_test_tag_dist_format'
        self.train()
        for tp in city2txt:
            txts = city2txt[tp]
            for csf, path in self.classifiers:
                pred_tags = csf.predict(txts)
                if not pred_tags or len(pred_tags) != len(txts):
                    raise Exception('Predict Results Exception')
                tag2dist = self.cal_tag_dist(pred_tags)
                ET.write_file(out_path, 'a', '%s,%s,%.4f,%.4f,%.4f\n' % (tp, len(txts), tag2dist['O'], tag2dist['P'], tag2dist['N']))
                if detail:
                    detail_path = 'city_test_simulate'
                    for tag, txt in zip(pred_tags, txts):
                        ET.write_file(detail_path, 'a', '%s -%s\n' % (tag, txt))

    def cal_tag_dist(self, pred_tags):
        tag2cnt = {'P':0, 'N':0, 'O':0}
        for tag in pred_tags:
            tag2cnt[tag] += 1
        tag2dist =  {tag:cnt * 1.0 / len(pred_tags) for tag,cnt in tag2cnt.items()}
        return tag2dist

    def train(self):
        for csf, path in self.classifiers:
            csf.train()

    def parse_topics_realtime(self):
        topic_cnt, total_cnt = 0, 0
        topic2txt = {}
        for name, txts in self.stats:
            for txt in txts:
                total_cnt += 1
                topic = ST.parse_topic(txt)
                if not topic:
                    continue
                topic_cnt += 1
                topic2txt.setdefault(topic, list())
                topic2txt[topic].append(txt)
        print 'total cnt: %s. topic stats cnt: %s' % (total_cnt, topic_cnt)
        print 'topic cnt: %s' % len(topic2txt)
        return topic2txt

    def parse_topics_test_data(self, in_path='../../test_data/topic_test_data'):
        if not os.path.exists(in_path):
            raise Exception('not exist path')
        topic2txt = {}
        with open(in_path, 'r') as f:
            for line in f:
                dic = json.loads(line.strip())
                tp, txt = dic.items()[0]
                topic2txt.setdefault(tp, list())
                topic2txt[tp].append(txt)
        linfo('func parse_topics_test_data-topic found: %s' % len(topic2txt))
        return topic2txt

    def parse_citys_test_data(self, in_path='../../test_data/city_test_data'):
        if not os.path.exists(in_path):
            raise Exception('not exist path')
        city2txt = {}
        with open(in_path, 'r') as f:
            for line in f:
                dic = json.loads(line.strip())
                tp, txt = dic.items()[0]
                city2txt.setdefault(tp,list())
                city2txt[tp].append(txt)
        total_cnt = len(city2txt)
        city2txt = {c:txt for c, txt in city2txt.items() if len(txt) > 500}
        linfo('func parse_city_test_data-city found: %s. stats count greater than threshold: %s' % (total_cnt, len(city2txt)))
        for city, txts in city2txt.items():
            print city, len(txts)
        return city2txt

            
    def load_stats(self):
        linfo('load files from dir %s' % self.stats_dir)
        files = [name for name in os.listdir(self.stats_dir) if os.path.isfile(os.path.join(self.stats_dir, name))]
        files = sorted(files)
        files = filter(lambda x: x > 'realtime_2016052620' and x < 'realtime_2016052714', files)
        linfo('stats files cnt: %s' % (len(files)))
        excludes = ['#', '【', '】']
        stats = []
        for name in files:
            tmp_p = os.path.join(self.stats_dir, name)
            with open(tmp_p, 'r') as f:
                stat = json.loads(f.readline().strip())
                #stat = filter(lambda x: all([case not in x for case in excludes]), stat)
                #linfo('file: %s. stat cnt: %s' % (name, len(stat)))
                stats.append((name, stat))
        linfo('stats files parse: %s' % (len(stats)))
        #name, stat = stats[0]
        #for x in stat:
        #    print x
        return stats

def main():
    obj = Simulator(['svm'])
    obj.run(detail=True)
    #obj.debug_topic_public_stats(detail=True)
    #obj.debug_city_public_stats()

if __name__ == '__main__':
    log_name = 'simulator.log'
    logging.basicConfig(filename='/home/lizhitao/log/%s' % log_name,format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise offline simulator')
    main()
    logging.info('end')
