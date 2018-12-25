#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy as np
import csv
import os
import config
import glob
import shutil

# 清空文件夹
shutil.rmtree(config.SEARCH_RESULT_VULSEEKER_DIR)
os.mkdir(config.SEARCH_RESULT_VULSEEKER_DIR)

filters = glob.glob(config.SEARCH_VULSEEKER_TFRECORD_DIR + os.sep + "*" + os.sep + "*.tfrecord")
for tfrecord in filters:
    cve_dir = tfrecord.split(os.sep)[-2]
    search_program = tfrecord.split(os.sep)[-1][:-9].split("__NUM__")[0]
    search_num = tfrecord.split(os.sep)[-1][:-9].split("__NUM__")[1].split("#")[0]
    cve_num = tfrecord.split(os.sep)[-1][:-9].split("__NUM__")[1].split("#")[1]
    os.system("./7_search_model_vulseeker.py '"+tfrecord+"' "+cve_dir+" "+cve_num+" "+search_num+" '"+search_program+"'")
print "finish!!!"
