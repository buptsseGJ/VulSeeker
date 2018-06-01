#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 处理重名文件

import config
import os
import subprocess
import glob
import shutil

'''
    srcfile='/Users/xxx/git/project1/test.sh'
    dstfile='/Users/xxx/tmp/tmp/1/test.sh'
'''


def mymovefile(srcfile, dstfile):
    if os.path.isfile(srcfile):
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        if os.path.exists(dstfile):
            os.remove(dstfile)
        shutil.move(srcfile, dstfile)  # 移动文件


for bin_path in config.STEP2_PORGRAM_ARR:
    # config.FEA_DIR + str(os.sep) + bin_path
    path = glob.glob(config.FEA_DIR + str(os.sep) + bin_path + os.sep + "*")
    for paths in path:
        temppath = paths + os.sep + 'temp'
        if os.path.exists(temppath):
            funset = set()
            filelist = os.listdir(temppath)
            for i in filelist:
                funset.add(i[0:-8])

            funfeacsv = open(paths + os.sep + 'functions_list_fea.csv')
            newfunfeacsv = open(paths + os.sep + 'functions_list.csv', 'w')
            while 1:
                line = funfeacsv.readline()
                if not line:
                    break
                else:
                    funname = line.split(',')[0]
                    if bin_path is "openssl" and funname in config.STEP2_CVE_OPENSSL_FUN_LIST.keys():
                        if funname in funset:
                            srcfile_pre = paths + os.sep + "temp" + os.sep + funname
                        else:
                            srcfile_pre = paths + os.sep + funname
                        dstfile_pre = config.CVE_FEATURE_DIR + os.sep + config.STEP2_CVE_OPENSSL_FUN_LIST.get(
                            funname) + os.sep + paths.split(os.sep)[-1] + os.sep + funname
                        mymovefile(srcfile_pre + "_fea.csv", dstfile_pre + "_fea.csv")
                        mymovefile(srcfile_pre + "_cfg.txt", dstfile_pre + "_cfg.txt")
                        mymovefile(srcfile_pre + "_dfg.txt", dstfile_pre + "_dfg.txt")
                    else:
                        if funname in funset:
                            # print "find"
                            dufunname = "duplicate_" + funname
                            newline = dufunname + line[len(funname):]
                            newfunfeacsv.write(newline)
                            print newline
                            # print line
                            srcfile_pre = temppath + os.sep + funname
                            dstfile_pre = paths + os.sep + dufunname
                            mymovefile(srcfile_pre + "_fea.csv", dstfile_pre + "_fea.csv")
                            mymovefile(srcfile_pre + "_cfg.txt", dstfile_pre + "_cfg.txt")
                            mymovefile(srcfile_pre + "_dfg.txt", dstfile_pre + "_dfg.txt")
                        else:
                            newfunfeacsv.write(line)
            newfunfeacsv.close()
            funfeacsv.close()
            # DELETE TEMP DIRECTORY
            shutil.rmtree(temppath)
            os.remove(paths + os.sep + 'functions_list_fea.csv')
        else:
            srcfile = paths + os.sep + 'functions_list_fea.csv'
            dstfile = paths + os.sep + 'functions_list.csv'
            mymovefile(srcfile, dstfile)



