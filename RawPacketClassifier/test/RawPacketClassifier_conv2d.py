from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import random
import pandas as pd

tf.set_random_seed(777) # for reproducibility

# hyperparameters
training_epochs = 1
batch_size = 100
learning_rate = 0.01

p = 0.01 # 1% of the datasets

# 1.Read dataset files
print('Reading Tor csv files...')
CHAT_gate_AIM_chat = pd.read_csv(r'tor/CHAT_gate_AIM_chat.csv', delimiter=',', dtype=np.float32, header=None, skiprows=lambda i: i>0 and random.random() > p)
VIDEO_Youtube_HTML5_Gateway = pd.read_csv(r'tor/VIDEO_Youtube_HTML5_Gateway.csv', delimiter=',', dtype=np.float32, header=None, skiprows=lambda i: i>0 and random.random() > p)

print('Reading nonTor csv files...')
facebook_chat = pd.read_csv(r'nonTor/facebook_chat.csv', delimiter=',', dtype=np.float32, header=None, skiprows=lambda i: i>0 and random.random() > p)
Email_IMAP_filetransfer = pd.read_csv(r'nonTor/Email_IMAP_filetransfer.csv', delimiter=',', dtype=np.float32, header=None, skiprows=lambda i: i>0 and random.random() > p)
#browsing2_2 = pd.read_csv(r'nonTor/browsing2-2.csv', delimiter=',', dtype=np.float32, header=None)

# 2.Split datasets into the training and testing sets (Shuffle is True by default)
print('Processing files... (train/test split + oject concatenation)')
CHAT_gate_AIM_chat_train, CHAT_gate_AIM_chat_test = train_test_split(CHAT_gate_AIM_chat, test_size=0.1)
VIDEO_Youtube_HTML5_Gateway_train, VIDEO_Youtube_HTML5_Gateway_test = train_test_split(VIDEO_Youtube_HTML5_Gateway, test_size=0.1)

facebook_chat_train, facebook_chat_test = train_test_split(facebook_chat, test_size=0.1)
Email_IMAP_filetransfer_train, Email_IMAP_filetransfer_test = train_test_split(Email_IMAP_filetransfer, test_size=0.1)
#browsing2_2_train, browsing2_2_test = train_test_split(browsing2_2, test_size=0.1)

# 3.Merge training and testing sets respectively.
concatenated_train_set = np.concatenate((CHAT_gate_AIM_chat_train, facebook_chat_train, VIDEO_Youtube_HTML5_Gateway_train, Email_IMAP_filetransfer_train), axis=0)
concatenated_test_set = np.concatenate((CHAT_gate_AIM_chat_test, facebook_chat_test, VIDEO_Youtube_HTML5_Gateway_test, Email_IMAP_filetransfer_test), axis=0)

concatenated_train_set = shuffle(concatenated_train_set)
concatenated_test_set = shuffle(concatenated_test_set)

x_train = concatenated_train_set[:, 0:-1]
y_train = concatenated_train_set[:, [-1]] # 0 ~ 15
train_length = x_train.shape[0]

x_test = concatenated_test_set[:, 0:-1]
y_test = concatenated_test_set[:, [-1]]
test_length = x_test.shape[0]

nb_classes = 16 # 0 ~ 15

# 4.Implement Network model
# input placeholders (108 nibbles(4-bits))
X = tf.placeholder(tf.float32, [None, 108])
X_re = tf.reshape(X, [-1, 9, 12, 1])   # reshaping as 9x12x1 like a image

# Tor -> P2P: 0 Browsing: 1, Email: 2, Chat: 3, Audio-Streaming: 4, Video-Streaming: 5, File Transfer: 6, VoIP: 7
# nonTor ->   8           9,        10,      11,                 12,                 13,               14,      15
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# L1 array in shape = (?, 9, 12, 1)
# Conv -> (?, 9, 12, 32)
# Pool -> (?, 5, 6, 32)
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X_re, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    W1_hist = tf.summary.histogram("weight1", W1)
    L1_hist = tf.summary.histogram("layer1", L1)

'''
Tensor("Conv2D:0", shape(?, 9, 12, 32), dtype=float32)
Tensor("Relu:0", shape(?, 9, 12, 32), dtype=float32)
Tensor("Maxpool:0", shape=(?, 5, 6, 32), dtype=float32)
'''

# L2 array in shape = (?, 5, 6, 32)
# Conv -> (?, 5, 6, 64)
# Pool -> (?, 3, 3, 64)
with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    L2_flat = tf.reshape(L2, [-1, 3 * 3 * 64])
    W2_hist = tf.summary.histogram("weight2", W1)
    L2_hist = tf.summary.histogram("layer2", L2)

'''
Tensor("Conv2D_1:0", shape=(?, 5, 6, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 5, 6, 64), dtype=float32)
Tensor("Maxpool_1:0", shape=(?, 3, 3, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 576), dtype=float32)
'''

# Final FC 3x3x64 inputs -> 16 outputs
with tf.name_scope("fully_connected_layer") as scope:
    W3 = tf.get_variable("W3", shape=[3 * 3 * 64, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.matmul(L2_flat, W3) + b
    W3_hist = tf.summary.histogram("weight3", W3)
    b_hist = tf.summary.histogram("biase", b)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# define cost/loss & optimizer
with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    cost_summ = tf.summary.scalar("cost", cost)

# 5.Train and Test the model
# initialize and launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/raw_packet_classification
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/conv2d")
    writer.add_graph(sess.graph)  # Show the graph
    sess.run(tf.global_variables_initializer())

    # train my model
    print('Learning started. It takes sometime...')
    for epoch in range(training_epochs):
        avg_cost = 0
        #total_batch = int(train_length / batch_size)
        for i in range(train_length):
            #batch_xs, batch_ys = ~~
            #feed_dict = {X: batch_xs, Y: batch_ys}
            summary, c, _ = sess.run([merged_summary, cost, optimizer], feed_dict={X: x_train, Y: y_train})
            writer.add_summary(summary, global_step=i)
            avg_cost += c / train_length

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

