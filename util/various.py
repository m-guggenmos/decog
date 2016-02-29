import datetime


def datetime_microseconds():
    return str(datetime.datetime.now()).replace('-', '').replace(' ', '').replace(':', '').replace('.', '')