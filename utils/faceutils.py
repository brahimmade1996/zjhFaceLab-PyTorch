
import time

def get_time(numricstr=False):
    timeformat='%Y-%m-%d %H:%M:%S'
    if numricstr:
        timeformat='%Y%m%d%H%M%S'
    return time.strftime(timeformat, time.localtime(time.time()))