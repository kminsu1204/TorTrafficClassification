import tensorflow as tf
import numpy as np
import random
import pandas as pd
from DatasetProcessor import Dataset

tf.set_random_seed(777) # for reproducibility

ds = Dataset(100, 0.2)  # sampling size must be smaller than 190 now (minimum size is 190)

# hyperparameters
training_epochs = 1
batch_size = 100
learning_rate = 0.03

class Model: 
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
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

            # L1 array in shape = (?, 108, 1)
            # Conv -> (?, 108, 16)
            # Pool -> (?, 54, 16)
            with tf.name_scope("layer1") as scope:
                W1 = tf.Variable(tf.random_normal([3, 1, 16], stddev=0.01))
                L1 = tf.nn.conv1d(X_re, W1, stride=1, padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.pool(L1, window_shape=[3], pooling_type='MAX', padding='SAME', strides=[2])
                W1_hist = tf.summary.histogram("weight1", W1)
                L1_hist = tf.summary.histogram("layer1", L1)

            # L2 array in shape = (?, 54, 16)
            # Conv -> (?, 54, 32)
            # Pool -> (?, 27, 32)
            with tf.name_scope("layer2") as scope:
                W2 = tf.Variable(tf.random_normal([3, 16, 32], stddev=0.01))
                L2 = tf.nn.conv1d(L1, W2, stride=1, padding='SAME')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.pool(L2, window_shape=[3], pooling_type='MAX', padding='SAME', strides=[2])
                L2_flat = tf.reshape(L2, [-1, 27 * 32])
                W2_hist = tf.summary.histogram("weight2", W1)
                L2_hist = tf.summary.histogram("layer2", L2)

            # Final FC 27x32 inputs -> 16 outputs
            with tf.name_scope("fully_connected_layer") as scope:
                W3 = tf.get_variable("W3", shape=[27 * 32, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.random_normal([nb_classes]))
                self.hypothesis = tf.matmul(L2_flat, W3) + b
                W3_hist = tf.summary.histogram("weight3", W3)
                b_hist = tf.summary.histogram("biase", b)
                hypothesis_hist = tf.summary.histogram("hypothesis", self.hypothesis)

        # define cost/loss & optimizer
        with tf.name_scope("cost") as scope:
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y_one_hot))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            cost_summ = tf.summary.scalar("cost", self.cost)
            correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y_one_hot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, x_train, y_train): # Train the model
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train})

    def get_accuracy(self, x_test, y_test): # Test the model
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})


# initialize and launch graph
with tf.Session() as sess:
    m1 = Model(sess, "m1")

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/conv1d")
    writer.add_graph(sess.graph)  # Show the graph

    sess.run(tf.global_variables_initializer())

    # train my model
    print('Learning started. It takes sometime...')
    for epoch in range(training_epochs):
        avg_cost = 0
        #total_batch = int(train_length / batch_size)
        for i in range(ds.train_length):
            #batch_xs, batch_ys = ~~
            #feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = m1.train(ds.x_train, ds.y_train)
            summary = sess.run([merged_summary], feed_dict={X: x_train, Y: y_train})
            writer.add_summary(summary, global_step=i)
            avg_cost += c / ds.train_length

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    print('Accuracy:', m1.get_accuracy(ds.x_test, ds.y_test))

