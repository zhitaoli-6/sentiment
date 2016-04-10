#encoding=utf-8
#!/usr/bin/env python
import sys, os
import json, logging, time, copy, random

from tool import EasyTool as ET
reload(sys)
sys.setdefaultencoding('utf-8')


linfo = logging.info

class NBClassifier(object):
    '''
    Base Class implment some common methods for classifiers
    '''
    class Word2Cnt(dict):
        def __init__(self):
            dict.__init__(self)
            #P:positive, N: negative, O: objective
            self["P"] = {}
            self["N"] = {}
            

    def __init__(self, train_data_path, **kwargs):
        self._path = train_data_path
        self._xs = []
        self._ys = []

    def predict(self, x):
        pass

    def train(self):
        #word2cnt = NBClassifier.Word2Cnt()
        
        self._load_data()
        self._train()

    def _train(self, shard_sz=10):
        rand_req = self._random_generate_shardlize(shard_sz)
        rid2shard = {}
        for i, rid in enumerate(rand_req):
            rid2shard.setdefault(rid, [])
            rid2shard[rid].append(i)

        rid2word_info, rid2tag_cnt = {}, {}
        total_word2cnt = NBClassifier.Word2Cnt()
        total_tag2cnt = {"P":0,"N":0,"O":0}
        for rid in range(1, shard_sz):
            shard = rid2shard[rid]
            rid2word_info[rid], rid2tag_cnt[rid] = self._cal_word2cnt(shard)
            for tag, w2c in rid2word_info[rid].items():
                for w, c in w2c.items():
                    total_word2cnt[tag].setdefault(w, 0)
                    total_word2cnt[tag][w] += c
            for tag, cnt in rid2tag_cnt.items():
                total_tag2cnt[tag] += cnt
        #cross_validation
        for rid in range(1, shard_sz):
            test_sd = rid2shard[rid]
            train_w2c = total_word2cnt
            train_t2c = total_tag2cnt
            test_w2c = rid2word_info[rid]
            test_t2c = rid2tag_cnt[rid]
            for tag, w2c in test_w2c.items():
                for w, c in w2c.items():
                    train_w2c[tag][w] -= c
            for tag, cnt in test_t2c.items():
                train_t2c[tag] -= cnt

            
            

            for tag, w2c in test_w2c.items():
                for w, c in w2c.items():
                    train_w2c[tag][w] += c
            for tag, cnt in test_t2c.items():
                train_t2c[tag] += cnt

    def _predict(self, x)

           
    
    def _cal_word2cnt(self, shard_indexs):
        word2cnt = NBClassifier.Word2Cnt()
        #word_total_cnt = 0
        tag2cnt = {"P":0,"N":0,"O":0}
        for index in shard_indexs:
            #word_total_cnt += len(x)
            x = self._xs[index] 
            tag = self._ys[index]
            tag2cnt[tag] += 1
            for w in x:
                word2cnt[tag].setdefault(w, 0)
                word2cnt[tag][w] += 1
        return word2cnt, tag2cnt

        #linfo("state average length: %.2f" % (word_total_cnt * 1.0 / len(self._xs)))
        #linfo('word cnt: %s' % len(self._word2cnt))
        #linfo('word average length: %.2f' % (word_total_cnt * 1.0 / len(self._word2cnt)))
        
        #cnt = 100
        #for i, w in enumerate(self._word2cnt.keys()):
        #    if i > cnt:
        #        break
        #    print '%s:%s' % (w, self._word2cnt[w])
        
    
    def _load_data(self):
        st = time.time()
        with open(self._path, 'r') as f:
            for line in f:
                dic = json.loads(line.strip())
                if len(dic) != 1:
                    print 'exception: %s' % line
                    continue
                tag, txt = dic.items()[0]
                self._ys.append(tag)
                self._xs.append(txt)
        linfo('time used: %.2f. instances cnt: %s' % (time.time() - st, len(self._xs)))

    def _random_generate_shardlize(self, shard_sz):
        if shard_sz <= 1:
            raise Exception('unvalid shard_sz for cross validation')
        return [random.randint(1, shard_sz) for i in range(len(self._xs))]
        

def main():
    #print dir(NBClassifier)
    nb = NBClassifier('stats/train_data')
    nb.train()
    
    
if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('begin supervise public states retriever')
    main()
    logging.info('end')
