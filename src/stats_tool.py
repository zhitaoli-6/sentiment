#encoding=utf-8 
#!/usr/bin/env python 
import sys, os
import json, logging, time, copy, random, math
from easy_tool import EasyTool as ET
from const import speical_province

reload(sys)
sys.setdefaultencoding('utf-8')

linfo = logging.info
ldebug = logging.debug

class StatsTool(object):
    
    debug_cnt = 0
    
    @staticmethod
    def load_data(in_path):
        linfo('load data from %s' % in_path)
        _ys, _xs = [], []
        st = time.time()
        with open(in_path, 'r') as f:
            for line in f:
                dic = json.loads(line.strip())
                if len(dic) != 1:
                    print 'exception: %s' % line
                    continue
                tag, txt = dic.items()[0]
                _ys.append(tag)
                _xs.append(txt)
        linfo('time used: %.2f. instances cnt: %s' % (time.time() - st, len(_xs)))
        return _xs, _ys


    #feature config: unigram, bigram, trigram, emoticon
    @classmethod
    def retrieve_feature(cls, txt, gram_icon_mixed=True, feature_extract_config=['unigram'], feature_representation_config='presence'):
        if feature_representation_config not in ['presence', 'frequency']:
            raise Exception('feature representation ERROR. not supported type for %s' % feature_representation_config)
        bags = []
        #depreciated
        if 'emoticon' in feature_extract_config:
            emoticons = cls._retrieve_emoticon(txt)
            for icon in emoticons:
                bags.append(icon)
        if not gram_icon_mixed:
            txt = cls._remove_emoticon(txt)
        for i, w in enumerate(txt):
            if 'unigram' in feature_extract_config:
                if w not in bags:
                    bags.append(w)
            if 'bigram' in feature_extract_config and i >= 1: 
                gram = '%s%s' % (txt[i-1], w)
                if gram not in bags:
                    bags.append(gram)
            if 'trigram' in feature_extract_config and i >= 2:
                gram = '%s%s%s' % (txt[i-2], txt[i-1], w)
                if gram not in bags:
                    bags.append(gram)
        return bags if feature_representation_config == 'frequency' else set(bags) 
 
    @classmethod
    def replace_url(cls, _xs, fill):
        linfo('begin replace url')
        urls = []
        for i, txt in enumerate(_xs):
            if 'http' in txt:
                ss = []
                ed = 0
                while 'http' in txt[ed:]:
                    txt = txt[ed:]
                    st = txt.find('http')
                    ss.append('%s' % txt[:st])
                    ss.append(fill)
                    ed = st
                    while (txt[ed] >= 'a' and txt[ed] <= 'z') or (txt[ed] >= 'A' and txt[ed] <= 'Z')  or (txt[ed] >= '0' and txt[ed] <= '9') or txt[ed] in ['.', ':', '/']:
                        ed += 1 
                        if ed >= len(txt):
                            break
                if ed < len(txt):
                    ss.append(txt[ed:])
                urls.append((i, ''.join(ss)))
        linfo('URL in %s instances replaced as H' % len(urls))
        for i, new_url in urls:
            _xs[i] = new_url
            #print new_url
        linfo('end replace url')
    @classmethod
    def replace_target(cls, _xs, fill):
        linfo('begin replace username')
        users = []
        for i, txt in enumerate(_xs):
            if '@' in txt:
                ss = []
                ed = 0
                while '@' in txt[ed:]:
                    txt = txt[ed:]
                    st = txt.find('@')
                    ss.append('%s' % txt[:st])
                    ss.append(fill)
                    ed = st
                    while ed < len(txt) and txt[ed] != ' ':
                        #print 'debug-ed:%s.%s' % (ed, txt[ed])
                        ed += 1 
                if ed < len(txt):
                    ss.append(txt[ed:])
                users.append((i, ''.join(ss)))
        linfo('USERNAME in %s instances replaced as T' % len(users))
        for i,new_user in users:
            _xs[i] = new_user
            #print _xs[i]

        linfo('end replace username')
    
    @classmethod
    def replace_topic(cls, _xs, fill):
        linfo('begin replace topic')
        cnt = 0
        for i, txt in enumerate(_xs):
            sharp_cnt = sum([c == '#' for c in txt])
            if sharp_cnt <= 1:
                continue
            elif sharp_cnt > 2:
                _xs[i] = ''
                cnt += 1
            else:
                cnt += 1
                tp = StatsTool.parse_topic(txt)
                _xs[i] = txt.replace(tp, fill)
        linfo('replace topic end. %s topic replaced' % (cnt))
   
    @classmethod
    def _retrieve_emoticon(cls, txt):
        icons = []
        if '[' not in txt or ']' not in txt:
            return icons
        ed = 0
        while '[' in txt[ed:]:
            txt = txt[ed:]
            st = txt.find('[')
            ed = st
            while ed < len(txt) and txt[ed] != ']':
                ed += 1
            if ed < len(txt) and txt[ed] == ']':
                icons.append(txt[st:ed+1])
            ed += 1
            if ed >= len(txt):
                break
        return icons

    @classmethod
    def remove_emoticon(cls, _xs):
        linfo('begin remove emoticon')
        for i in range(len(_xs)):
            _xs[i] = cls._remove_emoticon(_xs[i])
        linfo('end remove emoticon')
 
    @classmethod
    def _remove_emoticon(cls, txt):
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
        
    @classmethod 
    def remove_parenthesis(cls, _xs):
        logging.info("begin remove parenthesis")
        for i in range(len(_xs)):
            _xs[i] = _xs[i].replace('【', ' ').replace('】', ' ')
        logging.info("end remove parenthesis")
    
    @classmethod
    def random_shardlize(cls, shard_sz, rand_cnt, path='rand_req', save=False, load=False):
        if shard_sz <= 1:
            raise Exception('unvalid shard_sz for cross validation')
        if load:
            with open(path, 'r') as f:
                line = f.readline().strip()
                rand_req = map(int, line.split(' '))
                if len(rand_req) != rand_cnt:
                    raise Exception('Load rand_req fail. wrong results')
        else:
            rand_req =  [random.randint(1, shard_sz) for i in range(rand_cnt)]
        if save:
            ET.write_file(path, 'w', '%s\n'%' '.join(map(str, rand_req)))
            
        rid2shard = {}
        for i, rid in enumerate(rand_req):
            rid2shard.setdefault(rid, [])
            rid2shard[rid].append(i)
        return rid2shard
    
    @classmethod
    def parse_spatial(cls, dic):
        city = None
        if 'user' in dic and 'location' in dic['user']:
            locs = dic['user']['location'].split(' ') 
            p = locs[0]
            if p in speical_province:
                city = p
            elif len(locs) > 1:
                city =  dic['user']['location']
        return city
    @classmethod
    def parse_topic(cls, txt, sharp_threshold=None):
        topic = None
        st_i = txt.find('#')
        if st_i != -1:
            ed_i = txt[st_i+1:].find('#')
            if ed_i != -1:
                topic = txt[st_i : st_i + 1 + ed_i + 1]
                if sharp_threshold:
                    sharp_cnt = 2
                    for i in range(st_i + 1 + ed_i + 1, len(txt)):
                        if txt[i] == '#':
                            sharp_cnt += 1
                            if sharp_cnt > sharp_threshold:
                                return None
        return topic
    @staticmethod
    def load_raw_data(path, filter_enabled=True, row_num=None, replace_enabled=True):
        stat_ids = set()
        st = time.time()
        valid_stats = []
        redundant, retweet_cnt = 0, 0
        cur_row_num = 0
        with open(path, 'r') as f:
            for line in f:
                cur_row_num += 1
                if row_num != None:
                    if cur_row_num > row_num:
                        break
                dic = json.loads(line.strip())
                txt = dic['text']
                if filter_enabled:
                    if dic['id'] in stat_ids:
                        redundant += 1
                        continue
                    else:
                        stat_ids.add(dic['id'])
                    if '//' in txt:
                        st_i = txt.find('//')
                        if st_i == 0 or txt[st_i-1] != ':':
                            retweet_cnt += 1
                            continue
                valid_stats.append(txt)
        linfo('Load Raw Data-Filter: %s. Valid Stats Return: %s' % (filter_enabled, len(valid_stats)))
        if replace_enabled:
            StatsTool.replace_url(valid_stats, fill='H')
            StatsTool.replace_target(valid_stats, fill='T')
        return valid_stats

    @staticmethod
    def preprocess(txts):
        linfo('begin preprocess classifier input. cnt: %s' % (len(txts)))
        raw_txts = copy.copy(txts)
        StatsTool.replace_url(txts, 'H')
        StatsTool.replace_target(txts, 'T')
        StatsTool.replace_topic(txts, '')
        return raw_txts, txts
        #ret = filter(lambda x: x, txts)
        #valid_ids = [i for i, x in enumerate(txts) if x]
        #linfo('preprocess finish. cnt: %s' % (len(valid_ids)))
        #return valid_ids
        #return map(lambda i:raw_txts[i], valid_ids), map(lambda i:txts[i], valid_ids)



if __name__ == '__main__':
    st = u'@宁武发布 当选本期“网友点赞最多账号”'
    #StatsTool.replace_target(tmp, fill='T')
    demo = u'prefix#大连#, @tmp http://'
    txts = [demo]
    #StatsTool.remove_topic(txts, '')
    #print txts
    raw, txts = StatsTool.preprocess(txts)
    print raw
    print txts
