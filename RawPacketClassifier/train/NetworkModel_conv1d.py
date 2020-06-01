import tensorflow as tf
import numpy as np
import random
import pandas as pd
#from sklearn.metrics import precision_score, recall_score

tf.set_random_seed(777) # for reproducibility

class Model: 
    def __init__(self, sess, name, learning_rate):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self._build_net()

    def _build_net(self):   # Implement Network model
        with tf.variable_scope(self.name):
            # input placeholders (108 nibbles(4-bits))
            self.X = tf.placeholder(tf.float32, [None, 108])
            X_re = tf.reshape(self.X, [-1, 108, 1])

            # Tor -> P2P: 0 Browsing: 1, Email: 2, Chat: 3, Audio-Streaming: 4, Video-Streaming: 5, File Transfer: 6, VoIP: 7
            # nonTor ->   8           9,        10,      11,                 12,                 13,               14,      15
            nb_classes = 16 # for one hot encoding (0 ~ 15)
            self.Y = tf.placeholder(tf.int32, [None, 1])
            self.Y_one_hot = tf.one_hot(self.Y, nb_classes)  # one hot
            self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1, nb_classes])

            self.keep_prob = tf.placeholder(tf.float32)

            # L1 array in shape = (?, 108, 1)
            # Conv -> (?, 108, 32)
            # Pool -> (?, 54, 32)
            with tf.name_scope("cnn_layer1") as scope:
                W1 = tf.Variable(tf.random_normal([3, 1, 32], stddev=0.01))
                L1 = tf.nn.conv1d(X_re, W1, stride=1, padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.pool(L1, window_shape=[3], pooling_type='MAX', padding='SAME', strides=[2])
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
                
                self.W1_hist = tf.summary.histogram("weight", W1)
                self.L1_hist = tf.summary.histogram("layer", L1)

            # L2 array in shape = (?, 54, 32)
            # Conv -> (?, 54, 64)
            # Pool -> (?, 27, 64)
            with tf.name_scope("cnn_layer2") as scope:
                W2 = tf.Variable(tf.random_normal([3, 32, 64], stddev=0.01))
                L2 = tf.nn.conv1d(L1, W2, stride=1, padding='SAME')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.pool(L2, window_shape=[3], pooling_type='MAX', padding='SAME', strides=[2])
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
                L2_flat = tf.reshape(L2, [-1, 27 * 64])
                
                self.W2_hist = tf.summary.histogram("weight", W2)
                self.L2_hist = tf.summary.histogram("layer", L2)

            # L3 27x64 inputs -> 625 outputs
            with tf.name_scope("FC_layer3") as scope:
                W3 = tf.get_variable("W3", shape=[27 * 64, 625], initializer=tf.contrib.layers.xavier_initializer())
                b3 = tf.Variable(tf.random_normal([625]))
                L3 = tf.nn.relu(tf.matmul(L2_flat, W3) + b3)
                L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
                
                self.W3_hist = tf.summary.histogram("weight", W3)
                self.b3_hist = tf.summary.histogram("biase", b3)
                self.L3_hist = tf.summary.histogram("layer", L3)

            # L4 Final FC 625 inputs -> 16 outputs
            with tf.name_scope("Final_FC_layer4") as scope:
                W4 = tf.get_variable("W4", shape=[625, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
                b4 = tf.Variable(tf.random_normal([nb_classes]))
                self.hypothesis = tf.matmul(L3, W4) + b4
                
                self.W4_hist = tf.summary.histogram("weight", W4)
                self.b4_hist = tf.summary.histogram("biase", b4)
                self.hypothesis_hist = tf.summary.histogram("hypothesis", self.hypothesis)

        # define cost/loss & optimizer
        with tf.name_scope("scalar") as scope:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y_one_hot))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            
            cost_summ = tf.summary.scalar("cost", self.cost)
            
            actual = tf.argmax(self.Y_one_hot, 1)
            predicted = tf.argmax(self.hypothesis, 1)
            #import pdb;pdb.set_trace()
            '''self.TP = tf.count_nonzero(predicted * actual)
            self.TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            self.FP = tf.count_nonzero(predicted * (actual - 1))
            self.FN = tf.count_nonzero((predicted - 1) * actual)'''

            correct_prediction = tf.equal(predicted, actual)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            '''self.precision = tf.divide(self.TP,(self.TP + self.FP))
            self.recall = tf.divide(self.TP, (self.TP + self.FN))
            self.accuracy_by_calc = tf.divide((self.TP + self.TN), (self.TP + self.TN + self.FP + self.FN))'''

            #self.precision = precision_score(actual, predicted)
            #self.recall = recall_score(actual, predicted)

            _, self.precision_op = tf.metrics.precision(actual, predicted)
            _, self.recall_op = tf.metrics.recall(actual, predicted)
            _, self.tp_op = tf.metrics.true_positives(actual, predicted)
            #_, self.tn_op = tf.metrics.true_negatives(actual, predicted)
            _, self.fp_op = tf.metrics.false_positives(actual, predicted)
            _, self.fn_op = tf.metrics.false_negatives(actual, predicted)
            _, self.accuracy_op = tf.metrics.accuracy(actual, predicted)

            accuracy_summ = tf.summary.scalar("accuracy", self.accuracy)
            precision_summ = tf.summary.scalar("precision", self.precision_op)
            recall_summ = tf.summary.scalar("recall", self.recall_op)
            accuracy_op_summ = tf.summary.scalar("accuracy_op", self.accuracy_op)

            self.summary = tf.summary.merge_all()

    def train(self, x_train, y_train, keep_prob): # Train the model
        #print('x train: '+repr(x_train.shape)+', y train: '+repr(y_train.shape))
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.keep_prob: keep_prob})
    
    def train_summary(self, x_train, y_train, keep_prob): # Train the model and summarzie
        #print('x train: '+repr(x_train.shape)+', y train: '+repr(y_train.shape))
        return self.sess.run([self.cost, self.optimizer, self.summary], feed_dict={self.X: x_train, self.Y: y_train, self.keep_prob: keep_prob})
    
    def get_accuracy_precision_recall(self, x_test, y_test, keep_prob): # Test the model
        #import pdb;pdb.set_trace()
        #print('what?')
        return self.sess.run([self.accuracy, self.precision_op, self.recall_op, self.tp_op, self.fp_op, self.fn_op, self.accuracy_op], feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob})


