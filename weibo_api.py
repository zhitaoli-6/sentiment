#encoding=utf-8
import urllib2, json, sys

reload(sys)
sys.setdefaultencoding('utf-8')

def retrieve_new_msgs():
    url = 'https://api.weibo.com/2/statuses/public_timeline.json'
    urllib2.urlopen(url)
    return
    
    page = urllib2.urlopen(url).read()
    print page
   

def main():
    print 'begin retrieve newest msgs'
    retrieve_new_msgs()
if __name__ == '__main__':
    main()
