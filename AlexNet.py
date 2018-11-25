# Copyright (c) 2018 by huyz. All Rights Reserved.


import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST", one_hot=True)

learning_rate = 0.001
batch_size = 128
num_epochs = 2
dropout = 0.5

X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.variable_scope("conv1"):
    W1 = tf.get_variable(name="W1", shape=[11, 11, 1, 96], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable(name="b1", shape=[96], initializer=tf.zeros_initializer())
    W1_hist = tf.summary.histogram("W1", W1)
    b1_hist = tf.summary.histogram("b1", b1)
    conv1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")
    #conv1 = tf.layers.conv2d()
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)
    conv1_hist = tf.summary.histogram("conv1", conv1)

with tf.variable_scope("pool1"):
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("lrn1"):
    lrn1 = tf.nn.local_response_normalization(pool1, 5, bias=1.0,
                                            alpha=0.0001, beta=0.75)

with tf.variable_scope("conv2"):
    W2 = tf.get_variable(name="W2", shape=[5, 5, 96, 256], initializer=tf.truncated_normal_initializer())
    b2 = tf.get_variable(name="b2", shape=[256], initializer=tf.zeros_initializer())
    W2_hist = tf.summary.histogram("W1", W2)
    b2_hist = tf.summary.histogram("b2", b2)
    conv2 = tf.nn.conv2d(lrn1, W2, strides=[1, 1, 1, 1], padding="SAME")
    conv2 = tf.nn.bias_add(conv2, b2)
    conv2 = tf.nn.relu(conv2)
    conv2_hist = tf.summary.histogram("conv2", conv2)

with tf.variable_scope("pool2"):
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("lrn2"):
    lrn2 = tf.nn.local_response_normalization(pool2, 5, bias=1.0,
                                            alpha=0.0001, beta=0.75)

with tf.variable_scope("conv3"):
    W3 = tf.get_variable(name="W3", shape=[3, 3, 256, 384], initializer=tf.truncated_normal_initializer())
    b3 = tf.get_variable(name="b3", shape=[384], initializer=tf.truncated_normal_initializer())
    W3_hist = tf.summary.histogram("W3", W3)
    b3_hist = tf.summary.histogram("b3", b3)
    conv3 = tf.nn.conv2d(lrn2, W3, strides=[1, 1, 1, 1], padding="SAME")
    conv3 = tf.nn.bias_add(conv3, b3)
    conv3 = tf.nn.relu(conv3)
    conv3_hist = tf.summary.histogram("conv3", conv3)

with tf.variable_scope("conv4"):
    W4 = tf.get_variable(name="W4", shape=[3, 3, 384, 384], initializer=tf.truncated_normal_initializer())
    b4 = tf.get_variable(name="b4", shape=[384], initializer=tf.zeros_initializer())
    W4_hist = tf.summary.histogram("W4", W4)
    b4_hist = tf.summary.histogram("b4", b4)
    conv4 = tf.nn.conv2d(conv3, W4, strides=[1, 1, 1, 1], padding="SAME")
    conv4 = tf.nn.bias_add(conv4, b4)
    conv4 = tf.nn.relu(conv4)
    conv4_hist = tf.summary.histogram("conv4", conv4)

with tf.variable_scope("conv5"):
    W5 = tf.get_variable(name="W5", shape=[3, 3, 384, 256], initializer=tf.truncated_normal_initializer())
    b5 = tf.get_variable(name="b5", shape=[256], initializer=tf.zeros_initializer())
    W5_hist = tf.summary.histogram("W5", W5)
    b5_hist = tf.summary.histogram("b5", b5)
    conv5 = tf.nn.conv2d(conv4, W5, strides=[1, 1, 1, 1], padding="SAME")
    conv5 = tf.nn.bias_add(conv5, b5)
    conv5 = tf.nn.relu(conv5)
    conv5_hist = tf.summary.histogram("W1", conv5)

with tf.variable_scope("pool5"):
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("lrn5"):
    lrn5 = tf.nn.local_response_normalization(pool5, 5, bias=1.0,
                                            alpha=0.0001, beta=0.75)

with tf.variable_scope("fc1"):
    flatten = tf.reshape(lrn5, shape=[-1, 4*4*256])
    W6 = tf.get_variable(name="W6", shape=[4*4*256, 4096], initializer=tf.truncated_normal_initializer())
    b6 = tf.get_variable(name="b6", shape=[4096], initializer=tf.zeros_initializer())
    W6_hist = tf.summary.histogram("W6", W6)
    b6_hist = tf.summary.histogram("b6", b6)
    fc1 = tf.matmul(flatten, W6) + b6
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout)
    fc1_hist = tf.summary.histogram("W1", fc1)


with tf.variable_scope("fc2"):
    W7 = tf.get_variable(name="W7", shape=[4096, 4096], initializer=tf.truncated_normal_initializer())
    b7 = tf.get_variable(name="b7", shape=[4096], initializer=tf.zeros_initializer())
    W7_hist = tf.summary.histogram("W7", W7)
    b7_hist = tf.summary.histogram("b7", b7)
    fc2 = tf.matmul(fc1, W7) + b7
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=dropout)
    fc2_hist = tf.summary.histogram("fc2", fc2)

with tf.variable_scope("fc3"):
    W8 = tf.get_variable(name="W8", shape=[4096, 10], initializer=tf.truncated_normal_initializer())
    b8 = tf.get_variable(name="b8", shape=[10], initializer=tf.zeros_initializer())
    W8_hist = tf.summary.histogram("W8", W8)
    b8_hist = tf.summary.histogram("b8", b8)
    logit = tf.matmul(fc2, W8) + b8
    logit_hist = tf.summary.histogram("logit", logit)

# cost/loss function
with tf.variable_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y))
    cost_summ = tf.summary.scalar("loss", cost)

with tf.variable_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 准确率
is_correction= tf.equal(tf.argmax(Y, 1), tf.argmax(logit, 1))
accuracy = tf.reduce_mean(tf.cast(is_correction, tf.float32))
acc_summ = tf.summary.scalar("accuracy", accuracy)

# summary
summary = tf.summary.merge_all()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # 创建summary writer
    writer = tf.summary.FileWriter("./log")
    writer.add_graph(sess.graph)


    print("Learning start...")

    for epoch in range(num_epochs):
        avg_acc = 0
        avg_cost = 0
        num_batches = int(mnist.train.num_examples / batch_size)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_x, Y: batch_y}

        for i in range(num_batches):
            s, _, c, a = sess.run([summary, optimizer, cost, accuracy], feed_dict=feed_dict)
            writer.add_summary(s, global_step=i)
            avg_acc += a / num_batches
            avg_cost += c / num_batches
        print("Epoch: {}\tLoss: {:.9f}\tAccuracy: {:.3%}".format(epoch+1, avg_cost, avg_acc))
    print("Learning finished!")
