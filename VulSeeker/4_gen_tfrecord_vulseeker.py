#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np
import csv
import os
import time
import networkx as nx
import numba
import itertools
import config
# ===========  global parameters  ===========

T = 5  # iteration
N = 2  # embedding_depth
D = 8  # dimensional
P = 64  # embedding_size
B = 10  # mini-batch
lr = 0.0001  # learning_rate
# MAX_SIZE = 0 # record the max number of a function's block
epochs = 10
is_debug = True

#
data_folder = config.FEA_DIR
train_file = config.DATASET_DIR + os.sep + "train"+str(config.TRAIN_DATASET_NUM)+"_["+'_'.join(config.STEP3_PORGRAM_ARR)+"].csv"
test_file = config.DATASET_DIR + os.sep + "test"+str(config.TRAIN_DATASET_NUM)+"_["+'_'.join(config.STEP3_PORGRAM_ARR)+"].csv"
valid_file = config.DATASET_DIR + os.sep + "vaild"+str(config.TRAIN_DATASET_NUM)+"_["+'_'.join(config.STEP3_PORGRAM_ARR)+"].csv"

PREFIX = "_"+str(config.TRAIN_DATASET_NUM)+"_["+'_'.join(config.STEP3_PORGRAM_ARR)+"]"
TRAIN_TFRECORD = config.TFRECORD_VULSEEKER_DIR_DIR + os.sep + "train_"+PREFIX+".tfrecord"
TEST_TFRECORD = config.TFRECORD_VULSEEKER_DIR_DIR + os.sep + "test_"+PREFIX+".tfrecord"
VALID_TFRECORD = config.TFRECORD_VULSEEKER_DIR_DIR + os.sep + "valid_"+PREFIX+".tfrecord"

print TRAIN_TFRECORD

# ==================== load the function pairs list ===================
#       1.   load_dataset()      load the pairs list for learning, which are
#                                in train.csv, valid.csv, test.csv .
#       1-1. load_csv_as_pair()  process each csv file.
# =====================================================================
def load_dataset():
    """ load the pairs list for training, testing, validing
    """
    train_pair, train_label = load_csv_as_pair(train_file)
    valid_pair, valid_label = load_csv_as_pair(valid_file)
    test_pair, test_label = load_csv_as_pair(test_file)

    return train_pair, train_label, valid_pair, valid_label, test_pair, test_label

def load_csv_as_pair(pair_label_file):
    """ load each csv file, which record the pairs list for learning and its label ( 1 or -1 )
        csv file : uid, uid, 1/-1 eg: 1.1.128, 1.4.789, -1
        pair_dict = {(uid, uid) : -1/1}
    """
    pair_list = []
    label_list = []
    with open(pair_label_file, "r") as fp:
        pair_label = csv.reader(fp)
        for line in pair_label:
            pair_list.append([line[0], line[1]])
            label_list.append(int(line[2]))

    return pair_list, label_list

# =============== convert the real data to training data ==============
#       1.  construct_learning_dataset() combine the dataset list & real data
#       1-1. generate_adj_matrix_pairs()    traversal list and construct all the matrixs
#       1-1-1. convert_graph_to_adj_matrix()    process each cfg
#       1-2. generate_features_pair() traversal list and construct all functions' feature map
# =====================================================================
# @numba.jit
def construct_learning_dataset(uid_pair_list):
    """ Construct pairs dataset to train the model.
        attributes:
            adj_matrix_all  store each pairs functions' graph info, （i,j)=1 present i--》j, others （i,j)=0
            features_all    store each pairs functions' feature map
    """
    print "     start generate adj matrix pairs..."
    cfgs_1, cfgs_2, dfgs_1, dfgs_2 = generate_graph_pairs(uid_pair_list)

    print "     start generate features pairs..."
    ### !!! record the max number of a function's block
    feas_1, feas_2, max_size, num1, num2 = generate_features_pair(uid_pair_list)

    return cfgs_1, cfgs_2, dfgs_1, dfgs_2, feas_1, feas_2, num1, num2, max_size


# @numba.jit
def generate_graph_pairs(uid_pair_list):
    """ construct all the function pairs' cfg matrix.
    """
    cfgs_1 = []
    cfgs_2 = []
    dfgs_1 = []
    dfgs_2 = []
    # traversal all the pairs
    count = 0
    for uid_pair in uid_pair_list:
        if is_debug:
            count += 1
            print "         %04d cfg, [ %s , %s ]"%(count, uid_pair[0], uid_pair[1])
        adj_matrix_pair = []
        # each pair process two function
        graph_cfg = nx.read_adjlist(os.path.join(config.FEA_DIR, uid_pair[0]+"_cfg.txt"))
        # graph = uid_graph[uid_pair[0]]
        # origion_adj_1 = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        # origion_adj_1.resize(size, size, refcheck=False)
        # adj_matrix_all_1.append(origion_adj_1.tolist())
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        cfgs_1.append(",".join(list(itertools.chain.from_iterable(adj_str))))

        graph_dfg = nx.read_adjlist(os.path.join(data_folder, uid_pair[0]+"_dfg.txt"))
        graph= graph_dfg.copy()
        for node in graph.nodes():
            if not graph_cfg.has_node(node):
                graph_dfg.remove_node(node)
        graph_dfg.add_nodes_from(graph_cfg)
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_dfg, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        dfgs_1.append(",".join(list(itertools.chain.from_iterable(adj_str))))

        graph_cfg = nx.read_adjlist(os.path.join(data_folder, uid_pair[1]+"_cfg.txt"))
        # graph = uid_graph[uid_pair[1]]
        # origion_adj_2 = np.array(nx.convert_matrix.to_numpy_matrix(graph, dtype=float))
        # origion_adj_2.resize(size, size, refcheck=False)
        # adj_matrix_all_2.append(origion_adj_2.tolist())
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_cfg, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        cfgs_2.append(",".join(list(itertools.chain.from_iterable(adj_str))))

        graph_dfg = nx.read_adjlist(os.path.join(data_folder, uid_pair[1]+"_dfg.txt"))
        graph= graph_dfg.copy()
        for node in graph.nodes():
            if not graph_cfg.has_node(node):
                graph_dfg.remove_node(node)
        graph_dfg.add_nodes_from(graph_cfg)
        adj_arr = np.array(nx.convert_matrix.to_numpy_matrix(graph_dfg, dtype=float))
        adj_str = adj_arr.astype(np.string_)
        dfgs_2.append(",".join(list(itertools.chain.from_iterable(adj_str))))


    return cfgs_1, cfgs_2, dfgs_1, dfgs_2

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
        if is_debug:
            count += 1
            print "         %04d feature, [ %s , %s ]"%(count, uid_pair[0], uid_pair[1])

        node_vector = []
        block_feature_dic={}
        with open(os.path.join(config.FEA_DIR, uid_pair[0]+"_fea.csv"), "r") as fp:
            for line in csv.reader(fp):
                if line[0] == "":
                    continue
                # read every bolck's features
                block_feature = [float(x) for x in (line[8:15])]
                print line[0],block_feature
                # 删除某一列特征
                # del block_feature[6]
                block_feature_dic.setdefault(str(line[0]), block_feature)

        graph_cfg = nx.read_adjlist(os.path.join(data_folder, uid_pair[0] + "_cfg.txt"))
        for node in graph_cfg.nodes():
            node_vector.append(block_feature_dic[node])
        node_length.append(len(node_vector))
        num1.append(len(node_vector))
        node_arr = np.array(node_vector)
        node_str = node_arr.astype(np.string_)
        feas_1.append(",".join(list(itertools.chain.from_iterable(node_str))))


        node_vector = []
        block_feature_dic={}
        with open(os.path.join(data_folder, uid_pair[1]+"_fea.csv"), "r") as fp:
            for line in csv.reader(fp):
                if line[0] == "":
                    continue
                # read every bolck's features
                block_feature = [float(x) for x in (line[8:15])]
                # 删除某一列特征
                # del block_feature[6]
                block_feature_dic.setdefault(str(line[0]), block_feature)

        graph_cfg = nx.read_adjlist(os.path.join(data_folder, uid_pair[1] + "_cfg.txt"))
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

# ========================== the main function ========================
#       1.  load_dataset()  load the train, valid, test csv file.
#       2.  load_all_data() load the origion data, including block info, cfg by networkx.
#       3.  construct_learning_dataset() combine the csv file and real data, construct training dataset.
# =====================================================================
# 1. load the train, valid, test csv file.
data_time = time.time()
train_pair, train_label, valid_pair, valid_label, test_pair, test_label = load_dataset()
print "1. loading pairs list time", time.time() - data_time, "(s)"

# 2. load the origion data, including block info, cfg by networkx.
# graph_time = time.time()
# uid_cfg,uid_dfg, fea_dict = load_all_data()
# print "2. loading graph data time", time.time() - graph_time, "(s)"

# 3. construct training dataset.
cons_time = time.time()
# ======================= construct train data =====================
train_cfg_1, train_cfg_2,train_dfg_1, train_dfg_2,train_fea_1, train_fea_2, train_num1, train_num2, train_max \
    = construct_learning_dataset(train_pair)
# ========================== store in pickle ========================
node_list = np.linspace(train_max,train_max, len(train_label),dtype=int)
writer = tf.python_io.TFRecordWriter(TRAIN_TFRECORD)
for item1,item2,item3,item4,item5,item6, item7, item8, item9, item10 in itertools.izip(
        train_label, train_cfg_1, train_cfg_2, train_dfg_1, train_dfg_2, train_fea_1, train_fea_2,
        train_num1, train_num2, node_list):
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[item1])),
                'cfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                'cfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                'dfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                'dfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                'fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item6])),
                'fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item7])),
                'num1':tf.train.Feature(int64_list = tf.train.Int64List(value=[item8])),
                'num2':tf.train.Feature(int64_list = tf.train.Int64List(value=[item9])),
                'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item10]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()
# ========================== clean memory ========================
# del train_pair, train_adj_matrix_1,train_adj_matrix_2,train_feature_map_1,train_feature_map_2,train_max

# ======================= construct valid data =====================
valid_cfg_1, valid_cfg_2, valid_dfg_1, valid_dfg_2, valid_fea_1, valid_fea_2, valid_num1, valid_num2, valid_max \
    = construct_learning_dataset(valid_pair)
# ========================== store in pickle ========================
node_list = np.linspace(valid_max,valid_max, len(valid_label),dtype=int)
writer = tf.python_io.TFRecordWriter(VALID_TFRECORD)
for item1,item2,item3,item4,item5,item6, item7, item8, item9, item10 in itertools.izip(
        valid_label, valid_cfg_1, valid_cfg_2, valid_dfg_1, valid_dfg_2, valid_fea_1, valid_fea_2,
        valid_num1, valid_num2, node_list):
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[item1])),
                'cfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                'cfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                'dfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                'dfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                'fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item6])),
                'fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item7])),
                'num1':tf.train.Feature(int64_list = tf.train.Int64List(value=[item8])),
                'num2':tf.train.Feature(int64_list = tf.train.Int64List(value=[item9])),
                'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item10]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()
# ========================== clean memory ========================
# del valid_pair, valid_adj_matrix_1,valid_adj_matrix_2,valid_feature_map_1,valid_feature_map_2,valid_max

# ======================= construct test data =====================
test_cfg_1, test_cfg_2, test_dfg_1, test_dfg_2, test_fea_1, test_fea_2,test_num1, test_num2, test_max \
    = construct_learning_dataset(test_pair)
# ========================== store in pickle ========================
node_list = np.linspace(test_max,test_max, len(test_label),dtype=int)
writer = tf.python_io.TFRecordWriter(TEST_TFRECORD)
for item1,item2,item3,item4,item5,item6, item7, item8, item9, item10 in itertools.izip(
        test_label, test_cfg_1, test_cfg_2, test_dfg_1, test_dfg_2, test_fea_1, test_fea_2,
        test_num1, test_num2, node_list):
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[item1])),
                'cfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item2])),
                'cfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item3])),
                'dfg_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item4])),
                'dfg_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item5])),
                'fea_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item6])),
                'fea_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[item7])),
                'num1':tf.train.Feature(int64_list = tf.train.Int64List(value=[item8])),
                'num2':tf.train.Feature(int64_list = tf.train.Int64List(value=[item9])),
                'max': tf.train.Feature(int64_list=tf.train.Int64List(value=[item10]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()
