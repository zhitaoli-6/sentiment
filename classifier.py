#encoding=utf-8 
#!/usr/bin/env python
import sys, os
import json, logging, time, copy, random, math

from tool import EasyTool as ET
reload(sys)
sys.setdefaultencoding('utf-8')


linfo = logging.info
ldebug = logging.debug


#optimize: (1)laplace smoothing.

#prune: (2)url,(3)high and low frequency(low ability for classificaion), 
#optimize: (4)term presence feature perform better than bag of words
#10folder cross-validation-metric now(2, 3, 4 enabled): precision: 0.9065. recall: 0.9022.f-value:0.9044
#cross valdation without emoticon on test data set: precision: 0.6885. recall: 0.6863.f-value:0.6874

#FAIL: bigram, salience, entropy strategy

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

        #self._ngrams_config = ['unigram', 'bigram']
        self._ngrams_config = ['unigram', 'bigram']

    def predict(self, x):
        pass

    def train(self):
        #word2cnt = NBClassifier.Word2Cnt()
        
        #txt = '今天天气就是棒[哈哈] [太阳] [飞起来]#'
        #print self._remove_emoticon(txt)
        #return
        self._load_data()
        self._replace_url(fill=True)
        self._remove_emoticon()
        self._train()

    def _train(self, shard_sz=10):
        linfo('begin train classifier')
        rid2shard = self._random_shardlize(shard_sz,load=True)

        #rid2word_info = {}
        #total_word2cnt = NBClassifier.Word2Cnt()
        rid2tag_cnt, rid2word_presence = {}, {}
        total_word2presence = NBClassifier.Word2Cnt()
        total_tag2cnt = {"P":0,"N":0,"O":0}
        for rid in range(1, shard_sz+1):
            shard = rid2shard[rid]
            #rid2word_info[rid]
            rid2tag_cnt[rid], rid2word_presence[rid] = self._cal_shard2info(shard)
            #for tag, w2c in rid2word_info[rid].items():
            #    for w, c in w2c.items():
            #        total_word2cnt[tag].setdefault(w, 0)
            #        total_word2cnt[tag][w] += c
            for tag, w2p in rid2word_presence[rid].items():
                for w, c in w2p.items():
                    total_word2presence[tag].setdefault(w, 0)
                    total_word2presence[tag][w] += c
            for tag, cnt in rid2tag_cnt[rid].items():
                total_tag2cnt[tag] += cnt
        #self._debug_bigram(total_word2presence)
        #linfo('end')
        #return
        self._prune(total_word2presence, rid2word_presence, total_tag2cnt)
        #return
        #cross_validation
        linfo('beign cross validation')
        total_word2cnt = total_word2presence
        rid2word_info = rid2word_presence
        p, r, f = 0, 0, 0
        for rid in range(1, shard_sz+1):
            test_sd = rid2shard[rid]
            train_w2c,train_t2c  = total_word2cnt, total_tag2cnt
            test_w2c, test_t2c  = rid2word_info[rid], rid2tag_cnt[rid]
            self._reset_total_info(test_w2c, test_t2c, train_w2c, train_t2c, False)
            pred_cnt = {"P":0,"N":0,"O":0}
            for x in test_sd:
                predict_y = self._predict(x,train_w2c, train_t2c)
                if predict_y == self._ys[x]:
                    pred_cnt[self._ys[x]] += 1
                elif random.randint(1, 100) == 1:
                    ldebug('Predict Error-%s. Answer:%s. Predict:%s.' % (self._xs[x], self._ys[x], predict_y))
            precision = 1.0 * sum(pred_cnt.values()) / sum(test_t2c.values())
            calls = [pred*1.0/tag_cnt for pred, tag_cnt in zip(pred_cnt.values(), test_t2c.values()) if tag_cnt]
            if len(calls) != 2:
                raise Exception("only byclass is supported now! but %s tags are given" % len(calls))
            recall = sum(calls) / len(calls)
            f_value = 2*precision*recall / (precision + recall)
            p += precision
            r += recall
            f += f_value
            self._reset_total_info(test_w2c, test_t2c, train_w2c, train_t2c, True)
            #linfo('cross: precision: %.4f. recall: %.4f.f-value:%.4f' % (precision, recall, f_value))
        linfo('METRIC: precision: %.4f. recall: %.4f.f-value:%.4f' % (p / shard_sz, r / shard_sz, f / shard_sz))

    #predict test_data
    def _predict(self, index, train_w2c, train_t2c):
        p_pos, p_neg = (0.0, 0.0)
        txt = self._xs[index]
        grams = self._retrieve_ngrams(txt)
        for w in grams:
            p_pos += self._cal_likelihood(train_w2c["P"].get(w, 0), train_t2c["P"]) 
            p_neg += self._cal_likelihood(train_w2c["N"].get(w, 0), train_t2c["N"])  
        return "P" if p_pos > p_neg else "N"
           
    def _cal_likelihood(self, word_cnt, tag_cnt):
        return math.log(1.0 * word_cnt / tag_cnt + 1)

    def _reset_total_info(self, test_w2c, test_t2c, total_w2c, total_t2c, reset_flag):
        for tag, w2c in test_w2c.items():
            for w, c in w2c.items():
                if reset_flag:
                    total_w2c[tag][w] += c
                else:
                    total_w2c[tag][w] -= c
        for tag, cnt in test_t2c.items():
            if reset_flag:
                total_t2c[tag] += cnt
            else:
                total_t2c[tag] -= cnt

    #feature config: unigram, bigram, trigram
    def _cal_shard2info(self, shard_indexs):
        #word2cnt = NBClassifier.Word2Cnt()
        word2presence = NBClassifier.Word2Cnt() 
        #word_total_cnt = 0
        tag2cnt = {"P":0,"N":0,"O":0}
        for index in shard_indexs:
            #word_total_cnt += len(x)
            txt = self._xs[index] 
            tag = self._ys[index]
            tag2cnt[tag] += 1
            bags = self._retrieve_ngrams(txt)
            for w in bags:
                word2presence[tag].setdefault(w, 0)
                word2presence[tag][w] += 1
                #word2cnt[tag].setdefault(w, 0)
                #word2cnt[tag][w] += 1
            
        return tag2cnt, word2presence
    
    def _retrieve_ngrams(self, txt):
        bags = set()
        for i, w in enumerate(txt):
            if 'unigram' in self._ngrams_config:
                if w not in bags:
                    bags.add(w)
            if 'bigram' in self._ngrams_config and i >= 1:
                gram = '%s%s' % (txt[i-1], w)
                if gram not in bags:
                    bags.add(gram)
        return bags
    

    #prune those valueless words. Example:  url, words with high frequency
    def _prune(self, total_word2cnt, rid2word_info, tag2cnt):
        linfo('begin prune words:frequency now')
        #frequency prune
        upper_freq , lower_freq = 0.5, 0.000
        del_words = set()
        for tag, w2c in total_word2cnt.items():
            tag_cnt = tag2cnt[tag]
            for w, c in w2c.items():
                if c > tag_cnt * upper_freq:
                    del_words.add(w)
                    #print 'word: %s. high freq:%.4f in tag: %s' % (w, c*1.0/tag_cnt, tag)
                if c < tag_cnt * lower_freq:
                    del_words.add(w)
                    #print 'word: %s. low freq:%.4f in tag: %s' % (w, c*1.0/tag_cnt, tag)
        self.__delete_words(total_word2cnt, rid2word_info, del_words)

        #w2s = {}
        #pos_w2c, neg_w2c = total_word2cnt["P"], total_word2cnt["N"]
        #inter_words = set(pos_w2c.keys()) & set(neg_w2c.keys())
        #total_words = set(pos_w2c.keys()) | set(neg_w2c.keys())
        ##salience prune strategy
        ##for w in total_words:
        ##    p_w_pos = pos_w2c.get(w, 0) * 1.0 / tag2cnt['P']
        ##    p_w_neg = neg_w2c.get(w, 0) * 1.0 / tag2cnt['N']
        ##    w2s[w] = 1 - min(p_w_pos, p_w_neg) / (max(p_w_pos, p_w_neg) + 0.00001)
       
        ##entropy prune strategy
        #for w in total_words:
        #    w2s[w] = 0.0
        #    total_cnt = sum([w2c.get(w, 0) for tag,w2c in total_word2cnt.items()])
        #    if total_cnt == 0: 
        #        continue
        #    w2s[w] = sum([w2c[w]*1.0/total_cnt * math.log(w2c[w]*1.0/total_cnt) for tag, w2c in total_word2cnt.items() if w2c.get(w, 0) != 0])
        #sort_words = sorted(w2s.keys(), key=lambda x: w2s[x], reverse=True)
        #print 'pos_words:%s. neg_words:%s' % (len(pos_w2c), len(neg_w2c))
        #print 'len of inter_words for classes: %s' % len(inter_words)
        #for x in sort_words:
        #    print 'word: %s. score: %.4f' % (x, w2s[x])
        #self.__delete_words(total_word2cnt, rid2word_info, filter(lambda x:w2s[x] < -0.64, w2s.keys()))
        linfo('end prune')
 

    def __delete_words(self, total_word2cnt, rid2word_info, del_words):
        for tag, w2c in total_word2cnt.items():
            for w in del_words:
                if w in w2c:
                    del w2c[w]
        for rid, word_info in rid2word_info.items():
            for tag, w2c in word_info.items():
                for w in del_words:
                    if w in w2c:
                        del w2c[w]

    def _replace_url(self, fill=True):
        #URL replaced as '#'
        linfo('begin replace url')
        urls = []
        for i, txt in enumerate(self._xs):
            if 'http' in txt:
                ss = []
                ed = 0
                while 'http' in txt[ed:]:
                    txt = txt[ed:]
                    st = txt.find('http')
                    ss.append('%s' % txt[:st])
                    if fill:
                        ss.append('#')
                    ed = st
                    while (txt[ed] >= 'a' and txt[ed] <= 'z') or (txt[ed] >= 'A' and txt[ed] <= 'Z')  or (txt[ed] >= '0' and txt[ed] <= '9') or txt[ed] in ['.', ':', '/']:
                        ed += 1 
                        if ed >= len(txt):
                            break
                if ed < len(txt):
                    ss.append(txt[ed:])
                urls.append((i, ''.join(ss)))
        linfo('OPTIMIZATION:url in %s instances replaced' % len(urls))
        for i, new_url in urls:
            self._xs[i] = new_url
            #print new_url
        linfo('end replace url')

    def _remove_emoticon(self):
        linfo('begin remove emoticon')
        for i in range(len(self._xs)):
            self._xs[i] = self.__remove_emoticon(self._xs[i])
        linfo('end remove emoticon')

    def __remove_emoticon(self, txt):
        if '[' not in txt:
            return txt
        ss = []
        ed = 0
        while '[' in txt[ed:]:
            txt = txt[ed:]
            st = txt.find('[')
            ss.append(txt[:st])
            ed = st
            while ed < len(txt) and txt[ed] != ']':
                ed += 1
            ed += 1
        if ed < len(txt):
            ss.append(txt[ed:])
        return ''.join(ss)
        

    def _load_data(self):
        st = time.time()
        #cnt = 0
        with open(self._path, 'r') as f:
            for line in f:
                dic = json.loads(line.strip())
                if len(dic) != 1:
                    print 'exception: %s' % line
                    continue
                tag, txt = dic.items()[0]
                self._ys.append(tag)
                self._xs.append(txt)
                #cnt += 1
                #print txt, len(txt)
                #for i in range(len(txt)):
                #    print txt[i],
                #if cnt >= 1:
                #    break
        linfo('time used: %.2f. instances cnt: %s' % (time.time() - st, len(self._xs)))

    def _random_shardlize(self, shard_sz, save=False, load=False):
        if shard_sz <= 1:
            raise Exception('unvalid shard_sz for cross validation')
        if load:
            with open('rand_req', 'r') as f:
                line = f.readline().strip()
                rand_req = map(int, line.split(' '))
                if len(rand_req) != len(self._xs):
                    raise Exception('Load rand_req fail. wrong results')
        else:
            rand_req =  [random.randint(1, shard_sz) for i in range(len(self._xs))]
        if save:
            ET.write_file('rand_req', 'w', '%s\n'%' '.join(map(str, rand_req)))
            
        rid2shard = {}
        for i, rid in enumerate(rand_req):
            rid2shard.setdefault(rid, [])
            rid2shard[rid].append(i)
        return rid2shard
    
    def _debug_bigram(self, word2cnt):
        for tag, w2c in word2cnt.items():
            bigram = filter(lambda x: len(x) == 2, w2c.keys())
            for gram in bigram:
                if w2c[gram] > 10:
                    print 'tag: %s. gram: %s. cnt: %s' % (tag, gram, w2c[gram])

def main():
    #print dir(NBClassifier)
    #return
    nb = NBClassifier('stats/train_data')
    nb.train()
    
    
if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise classifier')
    main()
    logging.info('end')
