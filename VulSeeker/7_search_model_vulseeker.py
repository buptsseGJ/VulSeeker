#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np
import os
import config
import sys
import operator
# os.environ["PBR_VERSION"]='3.1.1'

# ===========  global parameters  ===========

T = 5  # iteration
N = 2  # embedding_depth
P = 64  # embedding_size
D = 16   # dimensional,feature num
B = 1 # mini-batch
lr = 0.0001  # learning_rate
# MAX_SIZE = 0 # record the max number of a function's block
max_iter = 100
decay_steps = 10 # 衰减步长
decay_rate = 0.0001 # 衰减率
snapshot = 1
is_debug = True

# =============== convert the real data to training data ==============
#       1.  construct_learning_dataset() combine the dataset list & real data
#       1-1. generate_adj_matrix_pairs()    traversal list and construct all the matrixs
#       1-1-1. convert_graph_to_adj_matrix()    process each cfg
#       1-2. generate_features_pair() traversal list and construct all functions' feature map
# =====================================================================
""" Parameter P = 64, D = 8, T = 7, N = 2,                  B = 10
     X_v = D * 1   <--->   8 * v_num * 10
     W_1 = P * D   <--->   64* 8    W_1 * X_v = 64*1
    mu_0 = P * 1   <--->   64* 1
     P_1 = P * P   <--->   64*64
     P_2 = P * P   <--->   64*64
    mu_2/3/4/5 = P * P     <--->  64*1
    W_2 = P * P     <--->  64*64
"""

def structure2vec(mu_prev, cdfg, x, name="structure2vec"):
    """ Construct pairs dataset to train the model.
    """
    with tf.variable_scope(name):
        # n层全连接层 + n-1层激活层
        # n层全连接层  将v_num个P*1的特征汇总成P*P的feature map
        # 初始化P1,P2参数矩阵，截取的正态分布模式初始化  stddev是用于初始化的标准差
        # 合理的初始化会给网络一个比较好的训练起点，帮助逃脱局部极小值（or 鞍点）
        W_1 = tf.get_variable('W_1', [D, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        P_CDFG_1 = tf.get_variable('P_CDFG_1', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        P_CDFG_2 = tf.get_variable('P_CDFG_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        L_CFG = tf.reshape(tf.matmul(cdfg, mu_prev, transpose_a=True), (-1, P))  # v_num * P
        S_CDFG =tf.reshape(tf.matmul(tf.nn.relu(tf.matmul(L_CFG, P_CDFG_2)), P_CDFG_1), (-1, P))

        return tf.tanh(tf.add(tf.reshape(tf.matmul(tf.reshape(x, (-1, D)), W_1), (-1, P)), S_CDFG))

def structure2vec_net(cdfgs, x, v_num):
    with tf.variable_scope("structure2vec_net") as structure2vec_net:
        B_mu_5 = tf.Variable(tf.zeros(shape = [0, P]), trainable=False)
        w_2 = tf.get_variable('w_2', [P, P], tf.float32, tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        for i in range(B):
            cur_size = tf.to_int32(v_num[i][0])
            # search = tf.slice(B_mu_0[i], [0, 0], [cur_size, P])
            mu_0 = tf.reshape(tf.zeros(shape = [2*cur_size, P]),(2*cur_size,P))
            cdfg = tf.slice(cdfgs[i], [0, 0], [cur_size*2, 2*cur_size])
            fea = tf.slice(x[i],[0,0], [cur_size*2,D])
            mu_1 = structure2vec(mu_0, cdfg, fea)  # , name = 'mu_1')
            structure2vec_net.reuse_variables()
            mu_2 = structure2vec(mu_1, cdfg, fea)  # , name = 'mu_2')
            mu_3 = structure2vec(mu_2, cdfg, fea)  # , name = 'mu_3')
            mu_4 = structure2vec(mu_3, cdfg, fea)  # , name = 'mu_4')
            mu_5 = structure2vec(mu_4, cdfg, fea)  # , name = 'mu_5')

            # B_mu_5.append(tf.matmul(tf.reshape(tf.reduce_sum(mu_5, 0), (1, P)), w_2))
            B_mu_5 = tf.concat([B_mu_5,tf.matmul(tf.reshape(tf.reduce_sum(mu_5, 0), (1, P)), w_2)],0)

        return B_mu_5

def cal_distance(model1, model2):
    a_b = tf.reduce_sum(tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(model1,(1,-1)),
                                                             tf.reshape(model2,(1,-1))],0),0),(B,P)),1,keep_dims=True)
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(model1),1,keep_dims=True))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(model2),1,keep_dims=True))
    distance = a_b/tf.reshape(tf.reduce_prod(tf.concat([tf.reshape(a_norm,(1,-1)),
                                                        tf.reshape(b_norm,(1,-1))],0),0),(B,1))
    return distance

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example, features={
        'cfg_1': tf.FixedLenFeature([], tf.string),
        'cfg_2': tf.FixedLenFeature([], tf.string),
        'dfg_1': tf.FixedLenFeature([], tf.string),
        'dfg_2': tf.FixedLenFeature([], tf.string),
        'fea_1': tf.FixedLenFeature([], tf.string),
        'fea_2': tf.FixedLenFeature([], tf.string),
        'num1': tf.FixedLenFeature([], tf.int64),
        'num2': tf.FixedLenFeature([], tf.int64),
        'max': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.string)})

    label = features['label']

    cfg_1 = features['cfg_1']
    cfg_2 = features['cfg_2']
    #adj_arr = np.reshape((adj_str.split(',')),(-1,D))
    #graph_1 = adj_arr.astype(np.float32)

    #adj_arr = np.reshape((adj_str.split(',')),(-1,D))
    #graph_2 = adj_arr.astype(np.float32)

    dfg_1 = features['dfg_1']
    dfg_2 = features['dfg_2']
    #adj_arr = np.reshape((adj_str.split(',')),(-1,D))
    #graph_1 = adj_arr.astype(np.float32)

    #adj_arr = np.reshape((adj_str.split(',')),(-1,D))
    #graph_2 = adj_arr.astype(np.float32)

    num1 = tf.cast(features['num1'], tf.int32)
    fea_1 = features['fea_1']
    #fea_arr = np.reshape((fea_str.split(',')),(node_num,node_num))
    #feature_1 = fea_arr.astype(np.float32)

    num2 =  tf.cast(features['num2'], tf.int32)
    fea_2 = features['fea_2']
    #fea_arr = np.reshape(fea_str.split(','),(node_num,node_num))
    #feature_2 = fea_arr.astype(np.float32)

    max_num = tf.cast(features['max'], tf.int32)

    return label, cfg_1, cfg_2, dfg_1, dfg_2, fea_1, fea_2, num1, num2, max_num


def get_batch( label, cfg_str1, cfg_str2, dfg_str1, dfg_str2, fea_str1, fea_str2, num1, num2, max_num):

    y = np.reshape(label, [B, 1])

    v_num_1 = []
    v_num_2 = []
    for i in range(B):
        v_num_1.append([int(num1[i])])
        v_num_2.append([int(num2[i])])

    # 补齐 martix 矩阵的长度
    cdfg_1 = []
    cdfg_2 = []
    for i in range(B):
        cfg_arr = np.array(cfg_str1[i].split(','))
        cfg_adj = np.reshape(cfg_arr, (int(num1[i]), int(num1[i])))
        cfg_ori1 = cfg_adj.astype(np.float32)
        cfg_ori1.resize(int(max_num[i]), int(max_num[i]), refcheck=False)
        dfg_arr = np.array(dfg_str1[i].split(','))
        dfg_adj = np.reshape(dfg_arr, (int(num1[i]), int(num1[i])))
        dfg_ori1 = dfg_adj.astype(np.float32)
        dfg_ori1.resize(int(max_num[i]), int(max_num[i]), refcheck=False)
        cdfg_zero = np.zeros([int(max_num[i]),int(max_num[i])])
        cdfg_cfg = np.concatenate([cfg_ori1, cdfg_zero], axis=1)
        cdfg_dfg = np.concatenate([cdfg_zero, dfg_ori1], axis=1)
        cdfg_vec1 = np.concatenate([cdfg_cfg, cdfg_dfg], axis=0)
        cdfg_1.append(cdfg_vec1.tolist())

        cfg_arr = np.array(cfg_str2[i].split(','))
        cfg_adj = np.reshape(cfg_arr, (int(num2[i]), int(num2[i])))
        cfg_ori2 = cfg_adj.astype(np.float32)
        cfg_ori2.resize(int(max_num[i]), int(max_num[i]), refcheck=False)
        dfg_arr = np.array(dfg_str2[i].split(','))
        dfg_adj = np.reshape(dfg_arr, (int(num2[i]), int(num2[i])))
        dfg_ori2 = dfg_adj.astype(np.float32)
        dfg_ori2.resize(int(max_num[i]), int(max_num[i]), refcheck=False)
        cdfg_zero = np.zeros([int(max_num[i]),int(max_num[i])])
        cdfg_cfg = np.concatenate([cfg_ori2, cdfg_zero], axis=1)
        cdfg_dfg = np.concatenate([cdfg_zero, dfg_ori2], axis=1)
        cdfg_vec2 = np.concatenate([cdfg_cfg, cdfg_dfg], axis=0)
        cdfg_2.append(cdfg_vec2.tolist())

    # 补齐 feature 列表的长度
    fea_1 = []
    fea_2 = []
    for i in range(B):
        fea_arr = np.array(fea_str1[i].split(','))
        fea_ori = fea_arr.astype(np.float32)
        fea_ori1 = np.resize(fea_ori,(np.max(v_num_1),D))
        fea_temp1 = np.concatenate([fea_ori1, fea_ori1], axis=1)
        fea_vec1 = np.resize(fea_temp1,(np.max(v_num_1)*2,D))
        fea_1.append(fea_vec1)

        fea_arr = np.array(fea_str2[i].split(','))
        fea_ori= fea_arr.astype(np.float32)
        fea_ori2 = np.resize(fea_ori,(np.max(v_num_2),D))
        fea_temp2 = np.concatenate([fea_ori2, fea_ori2], axis=1)
        fea_vec2 = np.resize(fea_temp2,(np.max(v_num_2)*2,D))
        fea_2.append(fea_vec2)

    return y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2

# 4.construct the network
# Initializing the variables
# Siamese network major part

# Initializing the variables

init = tf.global_variables_initializer()
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase=True)

v_num_left = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_left')
cdfg_left = tf.placeholder(tf.float32, shape=([B, None, None]), name='cdfg_left')
fea_left = tf.placeholder(tf.float32, shape=([B, None, D]), name='fea_left')

v_num_right = tf.placeholder(tf.float32, shape=[B, 1], name='v_num_right')
cdfg_right = tf.placeholder(tf.float32, shape=([B, None, None]), name='cdfg_right')
fea_right = tf.placeholder(tf.float32, shape=([B, None, D]), name='fea_right')

labels = tf.placeholder(tf.string, shape=([B, 1]), name='gt')

dropout_f = tf.placeholder("float")

with tf.variable_scope("siamese") as siamese:
    model1 = structure2vec_net(cdfg_left, fea_left, v_num_left)
    siamese.reuse_variables()
    model2 = structure2vec_net(cdfg_right, fea_right, v_num_right)

dis = cal_distance(model1, model2)

# loss = contrastive_loss(labels, dis)
# 
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

tfrecord_filename = sys.argv[1]
cve_dir = sys.argv[2]
cve_num = sys.argv[3]
search_num = sys.argv[4]
search_program = sys.argv[5]

list_search_label, list_search_cfg_1, list_search_cfg_2, list_search_dfg_1, list_search_dfg_2, list_search_fea_1, \
list_search_fea_2, list_search_num1, list_search_num2, list_search_max = read_and_decode(tfrecord_filename)
batch_search_label, batch_search_cfg_1, batch_search_cfg_2, batch_search_dfg_1, batch_search_dfg_2, batch_search_fea_1, \
batch_search_fea_2, batch_search_num1, batch_search_num2, batch_search_max  \
    = tf.train.batch([list_search_label, list_search_cfg_1, list_search_cfg_2, list_search_dfg_1, list_search_dfg_2,
                      list_search_fea_1, list_search_fea_2, list_search_num1, list_search_num2, list_search_max],
                     batch_size=B, capacity=B)
''''''

init_opt = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, config.MODEL_VULSEEKER_DIR+os.sep+config.STEP7_SEARCH_VULSEEKER_MODEL)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    total_batch = int(int(search_num) / int(cve_num))

    search_result = {}
    search_result_arr = {}
    for m in range(total_batch):
        predicts = 0
        predicts_arr = ""
        for version in range(int(cve_num)):
            search_label, search_cfg_1, search_cfg_2, search_dfg_1, search_dfg_2, search_fea_1, search_fea_2, \
            search_num1, search_num2, search_max \
                = sess.run([batch_search_label, batch_search_cfg_1, batch_search_cfg_2, batch_search_dfg_1,
                            batch_search_dfg_2, batch_search_fea_1, batch_search_fea_2, batch_search_num1,
                            batch_search_num2, batch_search_max])
            y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 \
                = get_batch(search_label,  search_cfg_1, search_cfg_2, search_dfg_1, search_dfg_2, search_fea_1,
                            search_fea_2, search_num1, search_num2,  search_max)
            predict = dis.eval(feed_dict={
                cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1, cdfg_right: cdfg_2,
                fea_right: fea_2, v_num_right: v_num_2, labels: y, dropout_f: 1.0})
            # print predict
            predicts = predicts + predict[0][0]
            predicts_arr = predicts_arr + str(predict[0][0])+","
        predicts = predicts / int(cve_num)
        print m,y[0][0],predicts
        search_result.setdefault(y[0][0],predicts)
        search_result_arr.setdefault(y[0][0],predicts_arr)

    search_result_sorted = sorted(search_result.items(),key=operator.itemgetter(1),reverse=True)
    print search_result_sorted
    if not os.path.exists(config.SEARCH_RESULT_VULSEEKER_DIR + os.sep + cve_dir):
        os.mkdir(config.SEARCH_RESULT_VULSEEKER_DIR + os.sep + cve_dir)
    result_file = config.SEARCH_RESULT_VULSEEKER_DIR + os.sep + cve_dir + os.sep + str(search_program) + ".csv"
    result_fp = open(result_file, 'w')
    for key, value in search_result_sorted:
        # predict_str = ""
        # for i in search_result_arr.get(key):
        #     predict_str = predict_str + predicts_arr[]
        result_fp.write(key.split("###")[0]+","+str(value)+ "," + key.split("###")[1] + "," + search_result_arr.get(key) +",\n")

    coord.request_stop()
    coord.join(threads)
