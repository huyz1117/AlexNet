# Copyright (c) 2018 by huyz. All Rights Reserved.

"""
AlexNet采用的结构如下：
1、Relu：标准CNN采用激活函数tanh,但进行梯度下降计算时，Relu函数训练速度更快
2、Local Response Normalization：局部归一化，在Relu之后采用，主要时为了提高模型泛化能力，现在通常采用BN
3、Overlapping Pool：重叠池化，传统CNN的pool层，ksize与stride相同，但论文发现，当stride<ksize时，
                    模型更难过拟合，所以论文采用ksize=3,stride=2
4、Dropout：Dropout层采用神经元丢弃策略，主要时为了避免模型过拟合，失活的神经元不再参与前向传播与反向传播
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#X_train = mnist.train.images
#print(X_train.shape)

X = tf.placeholder(tf.float32, shape=[None, 784])
X_img = tf.reshape(X, shape=[-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])

learning_rate = 0.001
num_epochs = 20
batch_size = 128
dropout = 0.5

######################## 定义卷积 ############################
def conv2d(name, x, W, b, strides=1, padding="SAME"):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)

####################### 定义池化 #############################
def maxpool(name, x, k=3, s=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                        padding=padding, name=name)

####################### 定义全连接 ############################
def fc(x, W, b):
    return tf.matmul(x, W) + b

# 规范化
def lrn(name, x, lsize=5):
    return tf.nn.local_response_normalization(x, lsize, bias=1.0, alpha=0.0001,
                        beta=0.75, name=name)

# 卷积核的参数
def weight(shape, mean=0.0, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev))

def bias(shape):
    return tf.Variable(tf.zeros(shape=shape))

weights = {"W1": weight([11, 11, 1, 96]),
            "W2": weight([5, 5, 96, 256]),
            "W3": weight([3, 3, 256, 384]),
            "W4": weight([3, 3, 384, 384]),
            "W5": weight([3, 3, 384, 256]),
            "W6": weight([4*4*256, 4096]),
            "W7": weight([4096, 4096]),
            "W8": weight([4096, 10])}

bias = {"b1": bias([96]),
        "b2": bias([256]),
        "b3": bias([384]),
        "b4": bias([384]),
        "b5": bias([256]),
        "b6": bias([4096]),
        "b7": bias([4096]),
        "b8": bias([10])}

def alexnet(x, weight, bias, dropout):

    ### 第1层卷积 ###
    ## conv size: 28*28*2 --> 28*28*96   ceil(28/1) ##
    conv1 = conv2d("conv1", x, weights["W1"], bias["b1"], padding="SAME")
    ## pool size: 28*28*96 --> 14*14*96  ceil(28/2) ##
    pool1 = maxpool("pool1", conv1, k=3, s=2, padding="SAME")
    ## lrn size: 14*14*96 --> 14*14*96  ##
    lrn1 = lrn("lrn1", pool1, lsize=5)

    ### 第2层卷积 ###
    ## conv size: 14*14*96 --> 14*14*256  ceil(14/1) ##
    conv2 = conv2d("conv2", lrn1, weights["W2"], bias["b2"], padding="SAME")
    ## pool size: 14*14*256 --> 7*7*256  ceil(14/2) ##
    pool2 = maxpool("pool2", conv2, k=3, s=2, padding="SAME")
    ## lrn size: 7*7*256 --> 7*7*256
    lrn2 = lrn("lrn2", pool2, lsize=5)

    ### 第3层卷积 ###
    ## conv size: 7*7*256 --> 7*7*384 ceil(7/1) ##
    conv3 = conv2d("conv3", lrn2, weights["W3"], bias["b3"], padding="SAME")

    ### 第4层卷积 ###
    ## conv size: 7*7*384 --> 7*7*384 ceil(7/1) ##
    conv4 = conv2d("conv4", conv3, weights["W4"], bias["b4"], padding="SAME")

    ### 第5层卷积 ###
    ## conv size: 7*7*384 --> 7*7*256 ceil(7/1) ##
    conv5 = conv2d("conv4", conv4, weights["W5"], bias["b5"], padding="SAME")
    ## pool size: 7*7*256 --> 4*4*256 ceil(7/2) ##
    pool5 = maxpool("pool5", conv5, k=3, s=2, padding="SAME")
    ## lrn size: 4*4256 --> 4*4*256 ##
    lrn5 = lrn("lrn5", pool5, lsize=5)

    ### 第1个全连接层 ###
    ## fc1 size: 4*4*256 --> 4096
    fc1 = tf.reshape(lrn5, shape=[-1, 4*4*256])
    fc1 = fc(fc1, weights["W6"], bias["b6"])
    fc1 = tf.nn.relu(fc1)
    ## dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    ### 第2个全连接层 ###
    ## fc2 size: 4096 --> 4096 ##
    fc2 = fc(fc1, weights["W7"], bias["b7"])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    ### 第3个全连接层 ###
    ## fc3 size: 4096 --> 10 ##
    logit = fc(fc2, weights["W8"], bias["b8"])

    return logit

########### 定义模型、损失函数、优化函数 ##############

### 定义模型 ###
pred = alexnet(X_img, weight, bias, dropout)

### 损失函数 ###
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

### 准确率 ###
is_correction= tf.equal(tf.argmax(Y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(is_correction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Learning start...")

    for epoch in range(num_epochs+1):
        avg_acc = 0
        avg_cost = 0
        num_batches = int(mnist.train.num_examples / batch_size)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_x, Y: batch_y}

        for i in range(num_batches):
            _, c, a = sess.run([optimizer, cost, accuracy], feed_dict=feed_dict)
            avg_acc += a / num_batches
            avg_cost += c / num_batches
        print("Epoch: {}\tLoss: {}\t Accuracy: {:.3%}".format(epoch+1, avg_cost, avg_acc))
    print("Learning finished!")
