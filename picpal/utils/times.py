# coding=utf-8
#
import datetime

##################################################
#  date and time processing utils
##################################################


def now():
    return datetime.datetime.now()


def today():
    return datetime.date.today()


def yesterday():
    return today() + datetime.timedelta(days=-1)


def tomorrow():
    return today() + datetime.timedelta(days=1)


def time_cost(begin_time):
    return (now() - begin_time).total_seconds()

