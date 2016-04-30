#encoding=utf-8 
#!/usr/bin/env python 
import sys, os
import json, logging, time, copy, random, math

from easy_tool import EasyTool as ET
from stats_tool import StatsTool as ST
reload(sys)
sys.setdefaultencoding('utf-8')


linfo = logging.info
ldebug = logging.debug

TAGS  = ['P', 'N', 'O']
def auto_tag(in_path='../stats/test_public_stats', out_path='../test_data/objective_test_data', tag_log='tagger.log'):
    start_line = 0
    if os.path.exists(tag_log):
        with open(tag_log, 'r') as f:
            line = f.readline()
            start_line = int(line)
    linfo('st_line: %s' % start_line)

    with open(in_path, 'r') as f:
        print 'please tag following states with "P:Positive", "N:Negative", "O:Objective"'
        for num, line in enumerate(f):
            if num < start_line:
                continue
            ET.write_file(tag_log, 'w', '%s' % (num+1))
            dic = json.loads(line.strip())
            txt = dic['text']
            if '#' in txt:
                continue
            print '--------------'
            print txt
            tag = raw_input()
            if tag in TAGS:
                item = {tag:txt}
                print '%s this state' % tag
                ET.write_file(out_path, 'a', '%s\n' % json.dumps(item))
            elif tag == 'Z':
                print 'exit'
                break
            else:
                print 'ignore this state'

#following is for check and statisticize test_data of auto_tag.
def auto_tag_check(in_path='../test_data/objective_test_data', out_path='../test_data/objective_test_data_final', tag_log='tagger_check.log'):
    start_line = 0
    if os.path.exists(tag_log):
        with open(tag_log, 'r') as f:
            line = f.readline()
            start_line = int(line)
    linfo('st_line: %s' % start_line)


    with open(in_path, 'r') as f:
        print 'please tag following states with "P:Positive", "N:Negative", "O:Objective"'
        for num, line in enumerate(f):
            if num < start_line:
                continue
            ET.write_file(tag_log, 'w', '%s' % (num+1))
            dic = json.loads(line.strip())
            tag, txt = dic.items()[0]
            if '#' in txt:
                continue
            print '--------------'
            print tag, txt
            tag = raw_input()
            if tag in TAGS:
                item = {tag:txt}
                print '%s this state' % tag
                ET.write_file(out_path, 'a', '%s\n' % json.dumps(item))
            elif tag == 'Z':
                print 'exit'
                break
            else:
                print 'ignore this state'
    
def visual_test(in_path='../test_data/objective_test_data', out_path='../test_data/visual_data'):
    if os.path.exists(out_path):
        os.system('rm %s' % out_path)
    #pp, nn = [], []
    lines = []
    with open(in_path, 'r') as f:
        for line in f:
            dic = json.loads(line.strip())
            tag, txt = dic.items()[0]
            new_line = '%s%s\n'  % (tag, txt)
            lines.append(new_line)
            #ET.write_file(out_path, 'a', new_line)
            #if tag == 'P':
            #    pp.append(new_line)
            #elif tag == 'N':
            #    nn.append(new_line)
    lines = sorted(lines)
    for line in lines:
        ET.write_file(out_path, 'a', line)
    

if __name__ == '__main__':
    logging.basicConfig(filename='/home/lizhitao/log/sentiment.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.INFO)
    logging.info('---------------------------\nbegin supervise tagger')
    #auto_tag()
    #auto_tag_check()
    #visual_test()
    while True:
        print ET.format_time()
        time.sleep(60)
    logging.info('end')
