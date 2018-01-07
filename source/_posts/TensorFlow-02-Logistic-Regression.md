---
title: 'TensorFlow_02: Logistic Regression'
date: 2018-01-07 18:25:21
tags: TensorFlow
categories: TensorFlow
comments: true
---

#### Logistic Regression using TensorFlow

```python
# logistic regression with tensorflow

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/home/cooli/Documents/DataSets/mnist", one_hot=True)

# print(mnist.train.num_examples)
# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)
#
# # -----------------------------
#
# print(mnist.test.num_examples)
# print(mnist.test.images.shape)
# print(mnist.test.labels.shape)

learning_rate = 0.01
batch_size = 128
epoches = 30

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32, [None, 10])

w = tf.Variable(initial_value=tf.zeros([784, 10], tf.float32))
b = tf.Variable(initial_value=tf.zeros([10], tf.float32))

logits = tf.matmul(x, w) + b

loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_ = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    num_batch = int(mnist.train.num_examples/batch_size)
    for i in range(epoches):
        total_loss = 0.
        for j in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, losses = sess.run([train_op, loss_], feed_dict={x: batch_x, y: batch_y})
            total_loss += losses
        print("epoch: {0}, loss: {1}".format(i+1, total_loss/batch_size))
    print("training done.")

    print("start testing")
    pred_y = tf.nn.softmax(logits)
    accuracy = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
    accuracy_ = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    print("accuracy on test set: ", sess.run([accuracy_], feed_dict={x: mnist.test.images, y: mnist.test.labels}))
```

xz

:)
