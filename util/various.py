import datetime


def datetime_microseconds():
    return str(datetime.datetime.now()).replace('-', '').replace(' ', '').replace(':', '').replace('.', '')

def elapsed_time(t):
    if t > 3*3600:
        return "%4.1fh" % (t / 3600.)
    elif t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.6fs" % (t)
