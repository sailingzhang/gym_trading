import logging
import sys
import  os
import datetime
from logging.handlers import RotatingFileHandler
def log_init(log_file,level):
    # logging.basicConfig(level=logging.DEBUG,
    #                     #format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    #                     format='%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s %(message)s',
    #                     # datefmt='%a, %d %b %Y %H:%M:%S.%f',
    #                     filename=log_file,
    #                     filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    #formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s %(message)s')

    handler = RotatingFileHandler(log_file, maxBytes=102400000, backupCount=5)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.setLevel(level)
    logger.addHandler(handler)

    console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)

    # logging.Formatter(fmt='%(asctime)s',datefmt='%Y-%m-%d,%H:%M:%S.%f')

