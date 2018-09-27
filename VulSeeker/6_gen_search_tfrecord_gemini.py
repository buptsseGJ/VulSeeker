#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 将所有的.o文件利用IDA 进行反汇编

import config
import os
import subprocess
import glob
import csv
import tensorflow as tf
import numpy as np
import networkx as nx
import itertools

# @numba.jit
def construct_learning_dataset(uid_pair_list):
    """ Construct pairs dataset to train the model.
        attributes:
            adj_matrix_all  store each pairs functions' graph info, （i,j)=1 present i--》j, others （i,j)=0
            features_all    store each pairs functions' feature map
    """
    print "     start generate adj matrix pairs..."
    cfgs_1, cfgs_2 = generate_graph_pairs(uid_pair_list)

    print "     start generate features pairs..."
    ### !!! record the max number of a function's block
    feas_1, feas_2, max_size, num1, num2 = generate_features_pair(uid_pair_list)

    return cfgs_1, cfgs_2, feas_1, feas_2, num1, num2, max_size


# @numba.jit
def generate_graph_pairs(uid_pair_list):
    """ construct all the function pairs' cfg matrix.
    """
    cfgs_1 = []
    cfgs_2 = []
    # traversal all the pairs
    count = 0
    for uid_pair in uid_pair_list:
        adj_matrix_pair = []
        # each pair process two function
        graph_cfg = nx.read_adjlist(os.path.join(config.CVE_FEATURE_DIR, uid_pair[0]+"_cfg.txt"))
        # graph = uid_graph[uid_pair[0]]
        # origion_adj_1 = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        # origion_adj_1.resize(size, size, refcheck=False)
        # adj_matrix_all_1.append(origion_adj_1.tolist())
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        cfgs_1.append(",".join(list(itertools.chain.from_iterable(adj_str))))

        graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, uid_pair[1]+"_cfg.txt"))
        # graph = uid_graph[uid_pair[1]]
        # origion_adj_2 = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        # origion_adj_2.resize(size, size, refcheck=False)
        # adj_matrix_all_2.append(origion_adj_2.tolist())
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        cfgs_2.append(",".join(list(itertools.chain.from_iterable(adj_str))))

    return cfgs_1, cfgs_2

# @numba.jit
def generate_features_pair(uid_pair_list):
    """ Construct each function pairs' block feature map.
    """
    feas_1 = []
    feas_2 = []
    num1 = []
    num2 = []
    node_length = []
    # traversal all the pairs
    count = 0
    for uid_pair in uid_pair_list:

        node_vector = []
        block_feature_dic={}
        with open(os.path.join(config.CVE_FEATURE_DIR, uid_pair[0]+"_fea.csv"), "r") as fp:
            for line in csv.reader(fp):
                if line[0] == "":
                    continue
                # read every bolck's features
                block_feature = [float(x) for x in (line[1:7])]
                print line[0],block_feature
                # 删除某一列特征
                # del block_feature[6]
                block_feature_dic.setdefault(str(line[0]), block_feature)

        graph_cfg = nx.read_adjlist(os.path.join(config.CVE_FEATURE_DIR, uid_pair[0] + "_cfg.txt"))
        for node in graph_cfg.nodes():
            node_vector.append(block_feature_dic[node])
        node_length.append(len(node_vector))
        num1.append(len(node_vector))
        node_arr = np.array(node_vector)
        node_str = node_arr.astype(np.string_)
        feas_1.append(",".join(list(itertools.chain.from_iterable(node_str))))


        node_vector = []
        block_feature_dic={}
        with open(os.path.join(config.FEA_DIR, uid_pair[1]+"_fea.csv"), "r") as fp:
            for line in csv.reader(fp):
                if line[0] == "":
                    continue
                # read every bolck's features
                block_feature = [float(x) for x in (line[1:7])]
                # 删除某一列特征
                # del block_feature[6]
                block_feature_dic.setdefault(str(line[0]), block_feature)

        graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, uid_pair[1] + "_cfg.txt"))
        for node in graph_cfg.nodes():
            node_vector.append(block_feature_dic[node])
        node_length.append(len(node_vector))
        num2.append(len(node_vector))
        node_arr = np.array(node_vector)
        node_str = node_arr.astype(np.string_)
        feas_2.append(",".join(list(itertools.chain.from_iterable(node_str))))

    num1_re = np.array(num1)
    num2_re = np.array(num2)
    #num1_re = num1_arr.astype(np.string_)
    #num2_re = num2_arr.astype(np.string_)
    return feas_1, feas_2, np.max(node_length),num1_re,num2_re


if not os.path.exists(config.SEARCH_DATASET_DIR):
    os.mkdir(config.SEARCH_DATASET_DIR)

cve_list = []
cve_filters = glob.glob(config.CVE_FEATURE_DIR + "\\*")
for cur_cve_dir in cve_filters:
    if os.path.isdir(cur_cve_dir):
        cve_list.append(cur_cve_dir.split(os.sep)[-1])

search_program_function_list = {}
for program in config.STEP6_SEARCH_PROGRAM_ARR:
    search_program_function_list[program] = {}
    tempdir = config.FEA_DIR + os.sep + str(program)
    filters = glob.glob(config.FEA_DIR + os.sep + str(program) + os.sep + "*")
    for i in filters:
        if os.path.isdir(i):
            search_program_function_list[program][i.split(os.sep)[-1]] = []
            search_list = i + os.sep + "functions_list.csv"
            with open(search_list, "r") as fp:
                for line in csv.reader(fp):
                    print line
                    if line[0] == "":
                        continue
                    search_program_function_list[program][i.split(os.sep)[-1]].append(line[0])
                    print line[0]

# 一个CVE 一个 tfrecord， 以CVE命名
for cur_cve in cve_list:
    search_pair_list = []
    for program,version  in search_program_function_list.items():
        for functions in version.values():
            print functions
            search_pair_list.append(config.CVE_FEATURE_DIR+os.sep+cur_cve,program+os.sep+version+os.sep+functions+",")
    search_cfg_1, search_cfg_2, search_fea_1, search_fea_2, search_num1, search_num2, search_max \
        = construct_learning_dataset(search_pair_list)
    node_list = np.linspace(search_max,search_max, len(search_pair_list),dtype=int)
    writer = tf.python_io.TFRecordWriter(config.SEARCH_GEMINI_TFRECORD + os.sep + cur_cve + ".tfrecord")
    for item1,item2,item3,item4,item5,item6, item7 in itertools.izip(
            search_cfg_1, search_cfg_2, search_fea_1, search_fea_2,
            search_num1, search_num2, node_list):
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'cfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item1])),
                    'cfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                    'fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                    'fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                    'num1':tf.train.Feature(int64_list = tf.train.Int64List(value=[item5])),
                    'num2':tf.train.Feature(int64_list = tf.train.Int64List(value=[item6])),
                    'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item7]))}))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()