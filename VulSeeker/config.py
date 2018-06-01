#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

# IDA Path

IDA32_DIR = "/media/yx/F/IDA/IDA_Pro_v6.4_Linux/idaq"
IDA64_DIR = "/media/yx/F/IDA/IDA_Pro_v6.4_Linux/idaq64"

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) #  The path of the current file
CODE_DIR = ROOT_DIR
O_DIR = ROOT_DIR+ os.sep + "0_Libs"                  #  The root path of All the binary file
IDB_DIR = ROOT_DIR+ os.sep + "0_Libs"                #  The root path of All the idb file
FEA_DIR = ROOT_DIR+ os.sep + "1_Features"            #  The root path of  the feature file
TFRECORD_VULSEEKER_DIR = ROOT_DIR+ os.sep + "3_TFRecord" + os.sep + "VulSeeker"
MODEL_VULSEEKER_DIR = ROOT_DIR+ os.sep + "4_Model" + os.sep + "VulSeeker"
CVE_FEATURE_DIR = ROOT_DIR+ os.sep + "5_CVE_Feature"
SEARCH_VULSEEKER_TFRECORD_DIR = ROOT_DIR+ os.sep + "6_Search_TFRecord" + os.sep + "VulSeeker"
SEARCH_RESULT_VULSEEKER_DIR = ROOT_DIR+ os.sep + "7_Search_Result" + os.sep + "VulSeeker"

# if convert all binary files into disassembly files
STEP1_GEN_IDB_FILE = True
STEP1_PORGRAM_ARR=["search_program"]#"openssl","coreutils","busybox","CVE-2015-1791"

# if extract feature file
STEP2_GEN_FEA = True
STEP2_PORGRAM_ARR=["search_program"]  #  all the project name. eg."openssl",,"busybox","coreutils","search_program"
STEP2_CVE_OPENSSL_FUN_LIST = {'ssl3_get_new_session_ticket':'CVE-2015-1791', 'OBJ_obj2txt':'CVE-2014-3508'}

STEP6_CVE_FUN_LIST = {'CVE-2015-1791':'ssl3_get_new_session_ticket', 'CVE-2014-3508':'OBJ_obj2txt'}
STEP6_GEN_SEARCH_VULSEEKER_TFRECORD = True
STEP6_SEARCH_PROGRAM_ARR = ["search_program"]

