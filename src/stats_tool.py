#encoding=utf-8 
#!/usr/bin/env python 
import sys, os
import json, logging, time, copy, random, math

reload(sys)
sys.setdefaultencoding('utf-8')


linfo = logging.info
ldebug = logging.debug

class StatsTool(object):
    
    debug_cnt = 0
    
    @staticmethod
    def load_data(in_path):
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
    def replace_url(cls, _xs, fill=True):
        #URL replaced as '#'
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
            _xs[i] = new_url
            #print new_url
        linfo('end replace url')

   
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
    def random_shardlize(cls, shard_sz, rand_cnt, save=False, load=False):
        if shard_sz <= 1:
            raise Exception('unvalid shard_sz for cross validation')
        if load:
            with open('rand_req', 'r') as f:
                line = f.readline().strip()
                rand_req = map(int, line.split(' '))
                if len(rand_req) != rand_cnt:
                    raise Exception('Load rand_req fail. wrong results')
        else:
            rand_req =  [random.randint(1, shard_sz) for i in range(len(rand_cnt))]
        if save:
            ET.write_file('rand_req', 'w', '%s\n'%' '.join(map(str, rand_req)))
            
        rid2shard = {}
        for i, rid in enumerate(rand_req):
            rid2shard.setdefault(rid, [])
            rid2shard[rid].append(i)
        return rid2shard
