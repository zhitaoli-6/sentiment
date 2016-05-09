#encoding=utf-8
#!/usr/bin/env python

import sys, json, logging, time, os, copy

from easy_tool import EasyTool as ET

reload(sys)
sys.setdefaultencoding('utf-8')
linfo = logging.info
ldebug = logging.debug
lexcept = logging.exception
write = ET.write_file

def visual_stats(in_path='stats/public_stats',out_path='stats/simple_public_stats'):
    '''
    Load public states and visualise
    '''
    lines = []
    with open(in_path, 'r') as f:
        for line in f:
            dic = json.loads(line.strip())
            lines.append('user: %s. created at: %s. text: %s\n' % (dic['user']['name'], dic['created_at'], dic['text']))
    for line in lines:
        ET.write_file(out_path, 'a', line)
    print 'cnt of lines: %s' % len(lines)

def parse_stats_emoticon(in_path='stats/public_stats', out_path='stats/emoticon_statistics'):
    '''
    Study emoticon info from public states
    '''
    st = time.time()
    icon2cnt = {}
    total_cnt, icon_line_cnt = (0, 0)
    with open(in_path, 'r') as f:
        for line in f:
            total_cnt += 1
            dic = json.loads(line.strip())
            txt = dic['text']
            id1 = txt.find('[')
            if id1 != -1:
                id2 = txt.find(']', id1)
                if id2 == -1: 
                    continue
                icon_line_cnt += 1
                icon  = txt[id1:id2+1]
                icon2cnt.setdefault(icon, 0)
                icon2cnt[icon] += 1
            #lines.append('user: %s. created at: %s. text: %s\n' % (dic['user']['name'], dic['created_at'], dic['text']))
    write = ET.write_file
    if os.path.exists(out_path):
        os.system('rm %s' % out_path)
    write(out_path, 'a', 'total_cnt: %s. emoticon_line_cnt: %s. time used: %.2f\n' % (total_cnt, icon_line_cnt, time.time() - st))
    icons = icon2cnt.keys()
    icons = sorted(icons, key=lambda x: icon2cnt[x], reverse=True)
    for icon in icons:
        cnt = icon2cnt[icon]
        write(out_path, 'a', '%s:%s\n' % (icon, cnt))

def load_emoticon(in_path='../stats/emoticon_selected'):
    '''
    Load selected emoticon
    '''
    pos_icons, neg_icons = [], []
    with open(in_path, 'r') as f:
        pos_flag = True
        for line in f:
            tks = line.strip().split(':')
            if len(tks) != 2:
                continue
            if tks[0] == '[泪]':
                pos_flag = False
            if pos_flag:
                pos_icons.append(tks[0])
            else:
                neg_icons.append(tks[0])
    return pos_icons, neg_icons

excludes = ['#']
def parse_emoticon_stats(in_path='stats/public_stats', out_path='stats/visual_train_data'):
    '''
    Parse states with selected emocicons.
    Dumps or visualise
    '''
    st = time.time()
    pos_icons, neg_icons = load_emoticon()

    icon2stat= {}
    total_cnt, icon_line_cnt = (0, 0)
    with open(in_path, 'r') as f:
        for line in f:
            total_cnt += 1
            dic = json.loads(line.strip())
            txt = dic['text']
            if any([x in txt for x in excludes]):
                continue
            pos_match = filter(lambda x: x in txt, pos_icons)
            neg_match = filter(lambda x: x in txt, neg_icons)
            if (pos_match and neg_match) or (not pos_match and not neg_match):
                continue
            if pos_match:
                for icon in pos_match:
                    icon2stat.setdefault(icon, [])
                    icon2stat[icon].append(txt)
                    break
            if neg_match:
                for icon in neg_match:
                    icon2stat.setdefault(icon, [])
                    icon2stat[icon].append(txt)
                    break

    write = ET.write_file
    if os.path.exists(out_path):
        os.system('rm %s' % out_path)
    pos_cnt = sum([len(icon2stat.get(x, [])) for x in pos_icons])
    neg_cnt = sum([len(icon2stat.get(x, [])) for x in neg_icons])
    icons = copy.copy(pos_icons)
    icons.extend(neg_icons)
    write(out_path, 'a', '----------------\ntotal_cnt: %s. pos_cnt: %s. neg_cnt: %s. time used: %.2fs\n' % (total_cnt, pos_cnt, neg_cnt, time.time()-st))
    for icon in icons:
        stats = icon2stat.get(icon, [])
        #write(out_path, 'a', '--------------------------------------\nicon: %s. stats_cnt: %s\n' % (icon, len(stats)))
        for stat in stats:
            #dic = {'%s' % ('P' if icon in pos_icons else 'N'):stat }
            #write(out_path, 'a', '%s\n' % json.dumps(dic))
            txt = '%s%s' % ('P' if icon in pos_icons else 'N', stat)
            write(out_path, 'a', '%s\n' % txt)
            #if icon in pos_icons:
            #    write(out_path, 'a', 'P%s\n' % stat)
            #else:
            #    write(out_path, 'a', 'N%s\n' % stat)

def load_news(news_path=['/home/lizhitao/repos/spider/data/cankao_records', '/home/lizhitao/repos/spider/data/people_news_records'], merge_path='../train_data/news_train_data'):
    news = set()
    st = time.time()
    for path in news_path:
        with open(path, 'r') as f:
            for line in f:
                tks = line.strip().split(',')
                if not tks or not tks[0]:
                    continue
                news.add(tks[0])
    linfo('new cnt: %s. time used: %.2f' % (len(news), time.time() - st))
    for new in news:
        dic = {'O':new}
        write(merge_path,'a','%s\n' % json.dumps(dic))

objective_excludes = ['[', ']', '#', '!', '?', '？','！']
objective_includes = ['【', '】', '空气质量指数']
def parse_objective_stats(in_path='../stats/public_stats', out_path='../stats/objective_train_data'):
    if os.path.exists(out_path):
        os.system('rm %s' % out_path)
    st = time.time()
    cnt = 1
    with open(in_path, 'r') as f:
        for line in f:
            dic = json.loads(line.strip())
            txt = dic['text']
            if any([x in txt for x in objective_excludes]):
                continue
            if any([x in txt for x in objective_includes]):
                dic = {'O':txt}
                ET.write_file(out_path, 'a', '%s\n' % json.dumps(dic))
                cnt += 1
            if cnt > 25000:
                break
    linfo('time used: %.2f. objective stats cnt: %s' % (time.time() - st, cnt))

def parse_topic_public_stats(in_path='../stats/train_public_stats',out_path='../test_data/topic_test_data'):
    st_t = time.time()
    topic_cnt, total_cnt = 0, 0
    topic2txt = {}
    with open(in_path, 'r') as f:
        for line in f:
            total_cnt += 1
            dic = json.loads(line.strip())
            txt = dic['text']
            st_i = txt.find('#')
            if st_i != -1:
                ed_i = txt[st_i+1:].find('#')
                if ed_i != -1:
                    topic_cnt += 1
                    topic = txt[st_i : st_i + 1 + ed_i + 1]
                    topic2txt.setdefault(topic, list())
                    topic2txt[topic].append(txt)
    topics = sorted(topic2txt.keys(), key=lambda x: len(topic2txt[x]), reverse=True)
    for t in topics:
        txts = topic2txt[t]
        if len(txts) > 7000:
            continue
        #print t, topic2txt[t]
        if len(txts) < 200:
            break
        for txt in txts:
            dic = {t:txt}
            ET.write_file(out_path, 'a', '%s\n' % json.dumps(dic))
        
    print 'total cnt: %s. topic stats cnt: %s' % (total_cnt, topic_cnt)
    print 'topic cnt: %s' % len(topic2txt)
    print 'time used: %.2f' % (time.time() - st_t)


speical_province = set([u'北京', u'上海', u'重庆',u'天津', u'台湾', u'澳门', u'香港'])
def parse_city_public_stats(in_path='../stats/train_public_stats', out_path='../test_data/city_test_data'):
    st_t = time.time()
    city2txt = {}
    city_stat_cnt, total_cnt = 0, 0
    stat_ids = set()
    txts_upperbound = 1000
    with open(in_path, 'r') as f:
        for line in f:
            total_cnt += 1
            dic = json.loads(line.strip()) 
            if dic['id'] in stat_ids:
                continue
            else:
                stat_ids.add(dic['id'])
            if 'user' in dic and 'location' in dic['user']:
                city_stat_cnt += 1
                locs = dic['user']['location'].split(' ') 
                p = locs[0]
                city = None
                if p in speical_province:
                    city = p
                elif len(locs) > 1:
                    city =  dic['user']['location']
                if not city:
                    continue
                city2txt.setdefault(city, list())
                if len(city2txt[city]) >= txts_upperbound:
                    continue
                city2txt[city].append(dic['text'])
    locs = sorted(city2txt.keys(), key=lambda x: len(city2txt[x]), reverse=True)
    print 'city cnt', len(city2txt)
    print 'city_stat_cnt', city_stat_cnt
    print 'total_cnt', total_cnt
    print 'time used: %.2f' % (time.time() - st_t)
    citys = sorted(city2txt.keys())
    #for x in citys:
    #    print x, len(city2txt[x])
    if os.path.exists(out_path):
        os.system('rm %s' % out_path)
    for x in locs:
        for txt in city2txt[x]:
            dic={x:txt}
            ET.write_file(out_path, 'a', '%s\n' % json.dumps(dic))

def main():
    #visual_stats()
    #parse_stats_emoticon()
    #parse_emoticon_stats()
    #load_news()
    #parse_objective_stats()
    #parse_topic_public_stats()
    parse_city_public_stats()
    
if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('begin supervise')
    main()
    logging.info('end')
