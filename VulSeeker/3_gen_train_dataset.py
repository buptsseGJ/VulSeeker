#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import csv
import os
import glob
import random
import config
## id,func_name,func_id,block_id_in_func,numeric constants,string constants,No. of transfer instructions,No. of calls,No. of instructinos,No. of arithmetic instructions,No. of logic instructions,No. of offspring,betweenness centrality
#生成训练集中函数block的数量
# block_num_min< block_num <= block_num_max
# if block_num_max = -1, 忽略这一设置,不考虑block数量
# block_num_min = 2
# block_num_max = 30
block_num_max = -1
block_num_min = -1
# PREFIX = "_coreutils_1000"

#正例负例数量
pos_num = 1
neg_num = 1

train_dataset_num = config.TRAIN_DATASET_NUM
test_dataset_num = int(train_dataset_num/10)
vaild_dataset_num = int(train_dataset_num/10)

func_list_file = config.DATASET_DIR + os.sep + "function_list"+str(config.TRAIN_DATASET_NUM)+"_["+ '_'.join(config.STEP3_PORGRAM_ARR)+"].csv"
train_file = config.DATASET_DIR + os.sep + "train"+str(config.TRAIN_DATASET_NUM)+"_["+'_'.join(config.STEP3_PORGRAM_ARR)+"].csv"
test_file = config.DATASET_DIR + os.sep + "test"+str(config.TRAIN_DATASET_NUM)+"_["+'_'.join(config.STEP3_PORGRAM_ARR)+"].csv"
vaild_file = config.DATASET_DIR + os.sep + "vaild"+str(config.TRAIN_DATASET_NUM)+"_["+'_'.join(config.STEP3_PORGRAM_ARR)+"].csv"

# function name, program name, block num, version uid list,
func_list_fp = open(func_list_file, "w")

index_uuid = dict()
index_count = 0
for program in config.STEP3_PORGRAM_ARR:
    tempdir = config.FEA_DIR + os.sep + str(program)
    filters = glob.glob(config.FEA_DIR + os.sep + str(program) + os.sep + "*")
    for i in filters:
        if os.path.isdir(i):
            index_uuid.setdefault(str(index_count), i.split(os.sep)[-1])
            print index_count
            index_count = index_count + 1

func_list_arr = []
func_list_dict = {}
for k,v in index_uuid.items():
    if not os.path.exists(config.FEA_DIR + os.sep + str(program) + os.sep + v + "/functions_list.csv"):
        continue
    with open(config.FEA_DIR + os.sep + str(program) + os.sep + v + "/functions_list_fea.csv", "r") as fp:
        print "gen_dataset : ",config.FEA_DIR + os.sep + str(program) + os.sep + v + "/functions_list.csv"
        for line in csv.reader(fp):
            print line
            if line[0] == "":
                continue
            if block_num_max > 0:
                if not ( int(line[1]) > block_num_min and int(line[1]) <= block_num_max ) :
                    continue
            if func_list_dict.has_key(line[0]):
                value = func_list_dict.pop(line[0])
                value = value + "," + line[4] + os.sep + line[5] + os.sep + line[0]
                func_list_dict.setdefault(line[0],value)
            else:
                #print line
                value = line[4] + os.sep + line[5] + os.sep + line[0]
                func_list_arr.append(line[0])
                func_list_dict.setdefault(line[0],value)


# coreutils6.5_mipsel64_gcc4.9_o0/ceil_lg,coreutils6.5_x64_gcc5.5_o0/ceil_lg,1
random.shuffle(func_list_arr)
func_list_test = []
func_list_train = []
func_list_vaild = []
for i in xrange(len(func_list_arr)):
    if i%12==0:
        func_list_test.append(func_list_arr[i])
    elif i%12==1:
        func_list_vaild.append(func_list_arr[i])
    else:
        func_list_train.append(func_list_arr[i])

train_fp = open(train_file, "w")
test_fp = open(test_file, "w")
vaild_fp = open(vaild_file, "w")

count = 0 #记录样本总量
cur_num = 0 #记录当前轮次 正例/负例 数量
while count < train_dataset_num:
    # 生成正例
    if cur_num < pos_num:
        random_func = random.sample(func_list_train,1)
        value = func_list_dict.get(random_func[0])
        select_list = value.split(',')
        if(len(select_list)<2):
            continue
        selected_list = random.sample(select_list,2)
        train_fp.write(selected_list[0]+","+selected_list[1]+",1\n")
    # 生成负例
    elif cur_num < pos_num + neg_num:
        random_func = random.sample(func_list_train,2)
        value1 = func_list_dict.get(random_func[0])
        select_list1 = value1.split(',')
        value2 = func_list_dict.get(random_func[1])
        select_list2 = value2.split(',')
        selected_list1 = random.sample(select_list1,1)
        selected_list2 = random.sample(select_list2,1)
        train_fp.write(selected_list1[0]+","+selected_list2[0]+",-1\n")
    cur_num += 1
    count += 1
    if cur_num == pos_num+neg_num:
        cur_num=0

count = 0 #记录样本总量
cur_num = 0 #记录当前轮次 正例/负例 数量
while count < test_dataset_num:
    # 生成正例
    if cur_num < pos_num:
        random_func = random.sample(func_list_test,1)
        value = func_list_dict.get(random_func[0])
        select_list = value.split(',')
        if(len(select_list)<2):
            continue
        selected_list = random.sample(select_list,2)
        test_fp.write(selected_list[0]+","+selected_list[1]+",1\n")
    # 生成负例
    elif cur_num < pos_num + neg_num:
        random_func = random.sample(func_list_test,2)
        value1 = func_list_dict.get(random_func[0])
        select_list1 = value1.split(',')
        value2 = func_list_dict.get(random_func[1])
        select_list2 = value2.split(',')
        selected_list1 = random.sample(select_list1,1)
        selected_list2 = random.sample(select_list2,1)
        test_fp.write(selected_list1[0]+","+selected_list2[0]+",-1\n")
    cur_num += 1
    count += 1
    if cur_num == pos_num+neg_num:
        cur_num=0

count = 0 #记录样本总量
cur_num = 0 #记录当前轮次 正例/负例 数量
while count < vaild_dataset_num:
    # 生成正例
    if cur_num < pos_num:
        random_func = random.sample(func_list_vaild,1)
        value = func_list_dict.get(random_func[0])
        select_list = value.split(',')
        if(len(select_list)<2):
            continue
        selected_list = random.sample(select_list,2)
        vaild_fp.write(selected_list[0]+","+selected_list[1]+",1\n")
    # 生成负例
    elif cur_num < pos_num + neg_num:
        random_func = random.sample(func_list_vaild,2)
        value1 = func_list_dict.get(random_func[0])
        select_list1 = value1.split(',')
        value2 = func_list_dict.get(random_func[1])
        select_list2 = value2.split(',')
        selected_list1 = random.sample(select_list1,1)
        selected_list2 = random.sample(select_list2,1)
        vaild_fp.write(selected_list1[0]+","+selected_list2[0]+",-1\n")
    cur_num += 1
    count += 1
    if cur_num == pos_num+neg_num:
        cur_num=0