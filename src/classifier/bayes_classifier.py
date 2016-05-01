#encoding=utf-8 
#!/usr/bin/env python 
import sys, os
import json, logging, time, copy, random, math

sys.path.append('/home/lizhitao/repos/sentiment/src')
from easy_tool import EasyTool as ET
from stats_tool import StatsTool as ST

reload(sys)
sys.setdefaultencoding('utf-8')


linfo = logging.info
ldebug = logging.debug

#optimize: (1)laplace smoothing.

#prune: (2)url replace,(3)remove words with too high and low frequency(low ability for classificaion), 
#optimize: (4)term presence feature perform better than bag of words. (5)unigam-bigram mixed perfrom better than unigram

#10folder cross-validation-metric now(2, 3, 4 enabled): precision: 0.9617. recall: 0.9595.f-value:0.9606
#cross valdation NO emoticon on test data set: precision: 0.7051. recall: 0.7037.f-value:0.7044
#manually tagged data: precision: 0.7878. recall: 0.7856.f-value:0.7867

#FAIL: salience, entropy strategy

#bug: emoticon. see file named 'case'

class BayesClassifier(object):
    '''
    Naive Bayes Model. Feature: unigram-bigram mixed
    '''
    class Word2Cnt(dict):
        def __init__(self):
            dict.__init__(self)
            #P:positive, N: negative, O: objective
            self["P"] = {}
            self["N"] = {}
            self["O"] = {}

    def __init__(self, classify_type='bi'):
        cc = classify_type
        if cc not in ['bi', 'tri']:
            raise Exception('INVALID CLASSIFIER TYPE')
        train_data = '../../train_data/%s_train_data' % cc
        test_data = '../../test_data/%s_test_data' % cc
        self.rand_path = '../rand/%s_rand_req' % cc 

        self._train_path = train_data
        config = ['unigram', 'bigram', 'trigram']

        '''
        ngrams_config: for feature extraction
        test_emoticon: to test emoticon influence
        '''
        self._ngrams_config = [config[0], config[1]]
        self._enable_test_emoticon = True

        linfo('classify type: %s' % cc)
        linfo('train feature extraction: %s' % self._ngrams_config)
        linfo('test emoticon: %s. \nend init bayes classifier' % self._enable_test_emoticon)
        self._test_path = test_data

    def predict(self, txt):
        if not hasattr(self, 'total_w2c') or not hasattr(self, 'total_t2c'):
            raise Exception('NOT TRAINED CLASSFIER')
        return self._predict(txt, self.total_w2c, self.total_t2c)

    def train(self, icon=True, cross=False):
        #word2cnt = BayesClassifier.Word2Cnt()
        
        #txt = '今天天气就是棒[哈哈] [太阳] [飞起来]#'
        #return
        #self._load_data()
        #self._replace_url(fill=True)
        self._train_xs, self._train_ys = ST.load_data(self._train_path)
        ST.replace_url(self._train_xs, fill=True)
        if not icon:
            ST.remove_emoticon(self._train_xs)
        self._train(cross_validation=cross)

    def _train(self, shard_sz=10, cross_validation=True):
        print self._ngrams_config
        linfo('begin train classifier')
        st = time.time()
        rid2shard = ST.random_shardlize(shard_sz, len(self._train_xs), load=True, path=self.rand_path)

        #rid2word_info = {}
        #total_word2cnt = BayesClassifier.Word2Cnt()
        rid2tag_cnt, rid2word_presence = {}, {}
        total_word2presence = BayesClassifier.Word2Cnt()
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
        self._prune(total_word2presence, rid2word_presence, total_tag2cnt)
        self.total_w2c, self.total_t2c = total_word2presence, total_tag2cnt
        linfo(self.total_t2c)
        #cross_validation
        if cross_validation:
            linfo('beign cross validation')
            p, r, f= self._cross_train(total_word2presence, rid2word_presence, total_tag2cnt, rid2tag_cnt, shard_sz, rid2shard)
            linfo('Classifier METRIC trained-precision: %.4f. recall: %.4f.f-value:%.4f. train cost used: %.2f' % (p , r , f, time.time()- st))
        else:
            linfo('beign train and test with manually tagged data set')
            p, r, f = self._all_train(total_word2presence, total_tag2cnt)
            linfo('Manually Tag Data Classifier METRIC trained-precision: %.4f. recall: %.4f.f-value:%.4f. train cost used: %.2f' % (p , r , f , time.time()- st))
        
    def _cross_train(self, total_word2cnt, rid2word_info, total_tag2cnt, rid2tag_cnt, shard_sz, rid2shard):
        p, r, f = 0, 0, 0
        for rid in range(1, shard_sz+1):
            n_st = time.time()
            test_sd = rid2shard[rid] 
            train_w2c,train_t2c  = total_word2cnt, total_tag2cnt
            test_w2c, test_t2c  = rid2word_info[rid], rid2tag_cnt[rid]
            self._reset_total_info(test_w2c, test_t2c, train_w2c, train_t2c, False)
            _s_xs = [self._train_xs[x] for x in test_sd]
            _s_ys = [self._train_ys[x] for x in test_sd]
            tp, tr, tf = self._batch_predict(_s_xs, _s_ys, train_w2c, train_t2c, test_t2c)
            p += tp
            r += tr
            f += tf
            self._reset_total_info(test_w2c, test_t2c, train_w2c, train_t2c, True)
            linfo('cross: precision: %.4f. recall: %.4f.f-value:%.4f. time uses this round: %.2f' % (tp, tr, tf, time.time() - n_st))
        return p/shard_sz, r/shard_sz, f/shard_sz

    def _all_train(self, total_word2cnt, total_tag2cnt):
        if os.path.exists(self._test_path):
            test_xs, test_ys = ST.load_data(self._test_path)
            #linfo('load manually tagged data count: %s' % len(test_xs))
        else:
            return
        test_t2c = {"P":0,"N":0,"O":0}
        for y in test_ys:
            if y not in test_t2c:
                raise Exception('Key Error in tag2cnt. unknown key: %s' % y)
            test_t2c[y] += 1
        print test_t2c
        return  self._batch_predict(test_xs, test_ys, total_word2cnt, total_tag2cnt, test_t2c)
        
 
    def _batch_predict(self, _xs, _ys, train_w2c, train_t2c, test_t2c):
        pred_cnt = {"P":0,"N":0,"O":0}
        n_st = time.time()
        for x, y in zip(_xs, _ys):
            predict_y = self._predict(x, train_w2c, train_t2c, emoticon=self._enable_test_emoticon)
            if predict_y == y:
                pred_cnt[y] += 1
            else:
                ldebug('Predict Error-%s. Answer:%s. Predict:%s.' % (x, y, predict_y))
        #linfo('Predict Success Cnt: %s/%s' % (sum(pred_cnt.values()), len(_xs)))
        precision = 1.0 * sum(pred_cnt.values()) / len(_xs)
        calls = [pred*1.0/tag_cnt for pred, tag_cnt in zip(pred_cnt.values(), test_t2c.values()) if tag_cnt]
        #if len(calls) != 2:
        #    raise Exception("only biclass is supported now! but %s tags are given" % len(calls))
        recall = sum(calls) / len(calls)
        f_value = 2*precision*recall / (precision + recall)
        print precision , recall, f_value
        return precision, recall, f_value

    #predict test_data
    def _predict(self, txt, train_w2c, train_t2c, debug=False, emoticon=True):
        #if emoticon and 'emoticon' not in self._ngrams_config:
        #    self._ngrams_config.append('emoticon')
        #elif not emoticon and 'emoticon' in self._ngrams_config:
        #    self._ngrams_config = filter(lambda x: x != 'emoticon', self._ngrams_config)
        #grams = self._retrieve_feature(txt)
        grams = ST.retrieve_feature(txt, feature_extract_config=self._ngrams_config, gram_icon_mixed=emoticon)
        if debug:
            linfo('begin debug case: %s' % txt)
        tag2score = {"P":0,"N":0,"O":0}
        for w in grams:
            for tag in tag2score:
                if not train_t2c[tag]:
                    continue
                score = self._cal_likelihood(train_w2c[tag].get(w, 0), train_t2c[tag]) 
                tag2score[tag] += score
                if debug:
                    linfo('DEBUG probability for gram %s when given tag %s is: %.4f. gram cnt: %s.tag cnt: %s' % (w, tag, score, train_w2c[tag].get(w, 0), train_t2c[tag]))
        pred_tag = sorted(tag2score.keys(), key=lambda x: tag2score[x], reverse=True)[0]
        if debug: 
            linfo('predict tag2score: %s' % tag2score)
        return pred_tag
           
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

    def _cal_shard2info(self, shard_indexs):
        #word2cnt = BayesClassifier.Word2Cnt()
        word2presence = BayesClassifier.Word2Cnt() 
        #word_total_cnt = 0
        tag2cnt = {"P":0,"N":0,"O":0}
        for index in shard_indexs:
            #word_total_cnt += len(x)
            txt = self._train_xs[index] 
            tag = self._train_ys[index]
            tag2cnt[tag] += 1
            bags = ST.retrieve_feature(txt, feature_extract_config=self._ngrams_config)
            for w in bags:
                word2presence[tag].setdefault(w, 0)
                word2presence[tag][w] += 1
                #word2cnt[tag].setdefault(w, 0)
                #word2cnt[tag][w] += 1
            
        return tag2cnt, word2presence

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
        #print 'total words cnt: %s' % len(total_words)
        #for w in total_words:
        #    print w
        
        ##salience prune strategy
        #for w in total_words:
        #    p_w_pos = pos_w2c.get(w, 0) * 1.0 / tag2cnt['P']
        #    p_w_neg = neg_w2c.get(w, 0) * 1.0 / tag2cnt['N']
        #    w2s[w] = 1 - min(p_w_pos, p_w_neg) / (max(p_w_pos, p_w_neg) + 0.00001)
       
        ###entropy prune strategy
        ##for w in total_words:
        ##    w2s[w] = 0.0
        ##    total_cnt = sum([w2c.get(w, 0) for tag,w2c in total_word2cnt.items()])
        ##    if total_cnt == 0: 
        ##        continue
        ##    w2s[w] = sum([w2c[w]*1.0/total_cnt * math.log(w2c[w]*1.0/total_cnt) for tag, w2c in total_word2cnt.items() if w2c.get(w, 0) != 0])
        #sort_words = sorted(w2s.keys(), key=lambda x: w2s[x], reverse=True)
        #print 'pos_words:%s. neg_words:%s' % (len(pos_w2c), len(neg_w2c))
        #print 'len of inter_words for classes: %s' % len(inter_words)
        #for x in sort_words:
        #    print 'word: %s. score: %.4f' % (x, w2s[x])
        #self.__delete_words(total_word2cnt, rid2word_info, filter(lambda x:w2s[x] < 0.1, w2s.keys()))
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

    def _debug_bigram(self, word2cnt):
        for tag, w2c in word2cnt.items():
            bigram = filter(lambda x: len(x) == 2, w2c.keys())
            for gram in bigram:
                if w2c[gram] > 10:
                    print 'tag: %s. gram: %s. cnt: %s' % (tag, gram, w2c[gram])

def main():
    #print dir(BayesClassifier)
    #return
    nb = BayesClassifier()
    nb.train(icon=True, cross=False)
    
if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise classifier')
    main()
    logging.info('end')
