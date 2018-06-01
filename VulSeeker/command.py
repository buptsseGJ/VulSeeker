#!/usr/bin/python
# -*- coding: UTF-8 -*-
import config
import os
import subprocess
import glob
import shutil
import sys

if config.STEP1_GEN_IDB_FILE:
    print "step1. convert to idb file"
    subprocess.call(["python", config.CODE_DIR+ os.sep + "1_gen_idb_file.py"])

if config.STEP2_GEN_FEA:
    print "step2-1. generate feature file"

    # if os.path.exists(config.FEA_DIR):
    #     shutil.rmtree(config.FEA_DIR)
    if not os.path.exists(config.FEA_DIR):
        os.mkdir(config.FEA_DIR)

    for program in config.STEP2_PORGRAM_ARR:
        tempdir = config.FEA_DIR + os.sep + str(program)
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)

        for version in os.listdir(config.IDB_DIR + os.sep + program):
            curFeaDir = config.FEA_DIR + str(os.sep) + str(program) + str(os.sep) + str(version)
            curBinDir = config.IDB_DIR + str(os.sep) + str(program) + str(os.sep) + str(version)
            if not os.path.exists(curFeaDir):
                os.mkdir(curFeaDir)
            filters = glob.glob(curBinDir + os.sep + "*.idb")
            filters = filters + (glob.glob(curBinDir + os.sep + "*.i64"))

            for i in filters:
                # function.id, block.id, instruction.id
                # idaq -S"2_gen_features.py C:\\Users\\yx\\Desktop\\dataset\\3_Featrue\\openssl\\openssl_1_0_1f_arm_o0 tty.idb program version"  tty.idb
                # print i
                if i.endswith("idb"):
                    print config.IDA32_DIR+" -S\""+config.CODE_DIR+ os.sep + "2_gen_features.py "+curFeaDir+"  "+i +"  "+ str(program) +"  "+ str(version) +"\"  "+i+"\n\n"
                    os.popen(config.IDA32_DIR+"   -S\""+config.CODE_DIR+ os.sep + "2_gen_features.py "+curFeaDir+" "+i +"  "+ str(program) +"  "+ str(version)  + "\"  "+i)
                    # subprocess.call(config.IDA32_DIR+" -A  -S\""+config.CODE_DIR+"\\2_gen_features.py "+curFeaDir+" "+i +"  "+ str(program) +"  "+ str(version)  + "\"  "+i )
                else:
                    print config.IDA64_DIR+" -S\""+config.CODE_DIR+ os.sep + "2_gen_features.py "+curFeaDir+"  "+i +"  "+ str(program) +"  "+ str(version) +"\"  "+i+"\n\n"
                    os.popen(config.IDA64_DIR+" -S\""+config.CODE_DIR+ os.sep + "2_gen_features.py "+curFeaDir+" "+i +"  "+ str(program) +"  "+ str(version)  + "\"  "+i)
                    # subprocess.call(config.IDA64_DIR+" -S\""+config.CODE_DIR+"\\2_gen_features.py "+curFeaDir+" "+i +"  "+ str(program) +"  "+ str(version)  + "\"  "+i )

    print "step2-2. process dump file"
    subprocess.call(["python", config.CODE_DIR + os.sep + "2_remove_duplicate.py"])

if config.STEP6_GEN_SEARCH_VULSEEKER_TFRECORD:
    subprocess.call(["python", config.CODE_DIR+ os.sep + "6_gen_search_tfrecord_vulseeker.py"])