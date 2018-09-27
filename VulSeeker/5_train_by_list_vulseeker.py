#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np
import csv
import time
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import config

# ===========  global parameters  ===========

T = 5  # iteration
N = 2  # embedding_depth
P = 64  # embedding_size
D = 8  # dimensional,feature num
B = 10 # mini-batch
lr = 0.0001  # learning_rate
# MAX_SIZE = 0 # record the max number of a function's block
max_iter = 100
decay_steps = 10 # 衰减步长
decay_rate = 0.0001 # 衰减率
snapshot = 1
is_debug = True

train_num = config.TRAIN_DATASET_NUM
valid_num = int(train_num/10)
test_num = int(train_num/10)
# PREFIX = "_coreutils_1000"
# TRAIN_TFRECORD="TFrecord/train_vulSeeker_data"+PREFIX+".tfrecord"
# TEST_TFRECORD="TFrecord/test_vulSeeker_data"+PREFIX+".tfrecord"
# VALID_TFRECORD="TFrecord/valid_vulSeeker_data"+PREFIX+".tfrecord"
PREFIX = "_"+str(config.TRAIN_DATASET_NUM)+"_["+'_'.join(config.STEP3_PORGRAM_ARR)+"]"
TRAIN_TFRECORD = config.TFRECORD_GEMINI_DIR + os.sep + "train_"+PREFIX+".tfrecord"
TEST_TFRECORD = config.TFRECORD_GEMINI_DIR + os.sep + "test_"+PREFIX+".tfrecord"
VALID_TFRECORD = config.TFRECORD_GEMINI_DIR + os.sep + "valid_"+PREFIX+".tfrecord"

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
            # test = tf.slice(B_mu_0[i], [0, 0], [cur_size, P])
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

def calculate_auc(labels, predicts):
    fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=1)
    AUC = auc(fpr, tpr)
    print "auc : ",AUC
    return AUC

def contrastive_loss(labels, distance):
    #    tmp= y * tf.square(d)
    #    #tmp= tf.mul(y,tf.square(d))
    #    tmp2 = (1-y) * tf.square(tf.maximum((1 - d),0))
    #    return tf.reduce_sum(tmp +tmp2)/B/2
    #    print "contrastive_loss", y,
    loss = tf.to_float(tf.reduce_sum(tf.square(distance - labels)))
    return loss


def compute_accuracy(prediction, labels):
    accu = 0.0
    threshold = 0.5
    for i in xrange(len(prediction)):
        if labels[i][0] == 1:
            if prediction[i][0] > threshold:
                accu += 1.0
        else:
            if prediction[i][0] < threshold:
                accu += 1.0
    acc = accu / len(prediction)
    return acc

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
        'label': tf.FixedLenFeature([], tf.int64),
        'cfg_1': tf.FixedLenFeature([], tf.string),
        'cfg_2': tf.FixedLenFeature([], tf.string),
        'dfg_1': tf.FixedLenFeature([], tf.string),
        'dfg_2': tf.FixedLenFeature([], tf.string),
        'fea_1': tf.FixedLenFeature([], tf.string),
        'fea_2': tf.FixedLenFeature([], tf.string),
        'num1': tf.FixedLenFeature([], tf.int64),
        'num2': tf.FixedLenFeature([], tf.int64),
        'max': tf.FixedLenFeature([], tf.int64)})

    label = tf.cast(features['label'], tf.int32)

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

labels = tf.placeholder(tf.float32, shape=([B, 1]), name='gt')

dropout_f = tf.placeholder("float")

with tf.variable_scope("siamese") as siamese:
    model1 = structure2vec_net(cdfg_left, fea_left, v_num_left)
    siamese.reuse_variables()
    model2 = structure2vec_net(cdfg_right, fea_right, v_num_right)

dis = cal_distance(model1, model2)

loss = contrastive_loss(labels, dis)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

list_train_label, list_train_cfg_1, list_train_cfg_2, list_train_dfg_1, list_train_dfg_2, list_train_fea_1, \
list_train_fea_2, list_train_num1, list_train_num2, list_train_max = read_and_decode(TRAIN_TFRECORD)
batch_train_label, batch_train_cfg_1, batch_train_cfg_2, batch_train_dfg_1, batch_train_dfg_2, batch_train_fea_1, \
batch_train_fea_2, batch_train_num1, batch_train_num2, batch_train_max  \
    = tf.train.batch([list_train_label, list_train_cfg_1, list_train_cfg_2, list_train_dfg_1, list_train_dfg_2,
                      list_train_fea_1, list_train_fea_2, list_train_num1, list_train_num2, list_train_max],
                     batch_size=B, capacity=10)

list_valid_label, list_valid_cfg_1, list_valid_cfg_2, list_valid_dfg_1, list_valid_dfg_2, list_valid_fea_1, \
list_valid_fea_2, list_valid_num1, list_valid_num2, list_valid_max = read_and_decode(VALID_TFRECORD)
batch_valid_label, batch_valid_cfg_1, batch_valid_cfg_2, batch_valid_dfg_1, batch_valid_dfg_2, batch_valid_fea_1, \
batch_valid_fea_2, batch_valid_num1, batch_valid_num2, batch_valid_max  \
    = tf.train.batch([list_valid_label, list_valid_cfg_1, list_valid_cfg_2, list_valid_dfg_1, list_valid_dfg_2,
                      list_valid_fea_1, list_valid_fea_2, list_valid_num1, list_valid_num2, list_valid_max],
                     batch_size=B, capacity=10)

list_test_label, list_test_cfg_1, list_test_cfg_2, list_test_dfg_1, list_test_dfg_2, list_test_fea_1, \
list_test_fea_2, list_test_num1, list_test_num2, list_test_max = read_and_decode(TEST_TFRECORD)
batch_test_label, batch_test_cfg_1, batch_test_cfg_2, batch_test_dfg_1, batch_test_dfg_2, batch_test_fea_1, \
batch_test_fea_2, batch_test_num1, batch_test_num2, batch_test_max  \
    = tf.train.batch([list_test_label, list_test_cfg_1, list_test_cfg_2, list_test_dfg_1, list_test_dfg_2,
                      list_test_fea_1, list_test_fea_2, list_test_num1, list_test_num2, list_test_max],
                     batch_size=B, capacity=10)
''''''
init_opt = tf.global_variables_initializer()
saver = tf.train.Saver()
#

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
# with tf.Session(config=config) as sess:
with tf.Session() as sess:
# with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init_opt)
    if config.SETP5_IF_RESTORE_VULSEEKER_MODEL:
        saver.restore(sess, config.MODEL_VULSEEKER_DIR + os.sep + config.STEP5_VULSEEKER_MODEL_TO_RESTORE)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Training cycle
    iter=0
    while iter < max_iter:
        iter += 1
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(train_num / B)
        start_time = time.time()
        # Loop over all batches
        # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_nu
        for i in range(total_batch):
            train_label, train_cfg_1, train_cfg_2, train_dfg_1, train_dfg_2, train_fea_1, train_fea_2, \
            train_num1, train_num2, train_max \
                = sess.run([batch_train_label, batch_train_cfg_1, batch_train_cfg_2, batch_train_dfg_1,
                            batch_train_dfg_2, batch_train_fea_1, batch_train_fea_2, batch_train_num1,
                            batch_train_num2, batch_train_max])
            y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 \
                = get_batch(train_label,  train_cfg_1, train_cfg_2, train_dfg_1, train_dfg_2, train_fea_1,
                            train_fea_2, train_num1, train_num2,  train_max)
            _, loss_value, predict = sess.run([optimizer, loss, dis], feed_dict = {
                cdfg_left: cdfg_1, fea_left: fea_1,v_num_left: v_num_1, cdfg_right: cdfg_2,fea_right: fea_2, v_num_right: v_num_2,labels: y, dropout_f: 0.9})
            tr_acc = compute_accuracy(predict, y)
            if is_debug:
                print '     %d    tr_acc %0.2f'%(i, tr_acc)
            avg_loss += loss_value
            avg_acc += tr_acc * 100
        duration = time.time() - start_time


        if iter%snapshot == 0:
            # validing model
            avg_loss = 0.
            avg_acc = 0.
            valid_start_time = time.time()
            for m in range(int(valid_num / B)):
                valid_label, valid_cfg_1, valid_cfg_2, valid_dfg_1, valid_dfg_2, valid_fea_1, valid_fea_2,  \
                valid_num1, valid_num2, valid_max \
                    = sess.run([batch_valid_label, batch_valid_cfg_1, batch_valid_cfg_2, batch_valid_dfg_1,
                                batch_valid_dfg_2,batch_valid_fea_1, batch_valid_fea_2, batch_valid_num1,
                                batch_valid_num2, batch_valid_max])
                y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 \
                    = get_batch(valid_label, valid_cfg_1, valid_cfg_2, valid_dfg_1, valid_dfg_2, valid_fea_1,
                                valid_fea_2,valid_num1, valid_num2,  valid_max)
                predict = dis.eval(feed_dict={
                    cdfg_left: cdfg_1, fea_left: fea_1, v_num_left: v_num_1, cdfg_right: cdfg_2,
                    fea_right: fea_2, v_num_right: v_num_2, labels: y, dropout_f: 0.9})
                tr_acc = compute_accuracy(predict, y)
                avg_loss += loss.eval(feed_dict={labels: y, dis: predict})
                avg_acc += tr_acc * 100
                if is_debug:
                    print '     tr_acc %0.2f'%(tr_acc)
            duration = time.time() - valid_start_time
            print 'valid set, %d,  time, %f, loss, %0.5f, acc, %0.2f' % (
                iter, duration, avg_loss / (int(valid_num / B)), avg_acc / (int(valid_num / B)))
            saver.save(sess, config.MODEL_VULSEEKER_DIR + os.sep + "vulseeker-model"+PREFIX+"_"+str(iter)+".ckpt")

            total_labels = []
            total_predicts = []
            avg_loss = 0.
            avg_acc = 0.
            test_total_batch = int(test_num / B)
            start_time = time.time()
            # Loop over all batches
            # get batch params label, graph_str1, graph_str2, feature_str1, feature_str2, num1, num2, max_num
            for m in range(test_total_batch):
                test_label, test_cfg_1, test_cfg_2, test_dfg_1, test_dfg_2, \
                test_fea_1, test_fea_2, test_num1, test_num2, test_max = sess.run(
                    [batch_test_label, batch_test_cfg_1, batch_test_cfg_2, batch_test_dfg_1, batch_test_dfg_2,
                     batch_test_fea_1,batch_test_fea_2, batch_test_num1, batch_test_num2, batch_test_max])
                y, cdfg_1, cdfg_2, fea_1, fea_2, v_num_1, v_num_2 \
                    = get_batch(test_label, test_cfg_1, test_cfg_2, test_dfg_1, test_dfg_2,
                                test_fea_1, test_fea_2, test_num1, test_num2, test_max)
                predict = dis.eval(
                    feed_dict={cdfg_left: cdfg_1, fea_left: fea_1,v_num_left: v_num_1, cdfg_right: cdfg_2,
                fea_right: fea_2, v_num_right: v_num_2,labels: y, dropout_f: 1.0})
                tr_acc = compute_accuracy(predict, y)
                avg_loss += loss.eval(feed_dict={labels: y, dis: predict})
                avg_acc += tr_acc * 100
                total_labels.append(y)
                total_predicts.append(predict)
                if is_debug:
                    print '     %d    tr_acc %0.2f' % (m, tr_acc)
            duration = time.time() - start_time
            total_labels = np.reshape(total_labels, (-1))
            total_predicts = np.reshape(total_predicts, (-1))
            print calculate_auc(total_labels, total_predicts)
            print 'test set, time, %f, loss, %0.5f, acc, %0.2f' % (duration, avg_loss / test_total_batch, avg_acc / test_total_batch)

# 保存模型
    saver.save(sess, config.MODEL_VULSEEKER_DIR + os.sep + "vulseeker-model"+PREFIX+"_final.ckpt")

    coord.request_stop()
    coord.join(threads)