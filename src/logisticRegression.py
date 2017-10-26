import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/data/mnist', one_hot=True)
print(mnist.train.labels.shape)
print(mnist.train.images.shape)

batch_size = 128
X = tf.placeholder(tf.float32, [batch_size, 784], name="X")
Y = tf.placeholder(tf.int32, [batch_size, 10], name="Y")

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weight")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="loss")

loss = tf.reduce_mean(entropy)
learning_rate = 0.01

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("C:/Temp/log/logistic", sess.graph)
    writer.close()
    start_time = time.time()
    n_batches = int(mnist.train.num_examples / batch_size)

    for i in range(30):
        total_loss = 0

        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
            # print('average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print('Total time:{0} seconds'.format(time.time() - start_time))
    print("done")

    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch[0]

    print("Accuracy: {0}".format(total_correct_preds / mnist.test.num_examples))
