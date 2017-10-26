import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# prepare input data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)
print(mnist.train.labels.shape)
print(mnist.train.images.shape)

# prepare placeholder for data
X = tf.placeholder(tf.float32, [None, 784], name="X")
Y = tf.placeholder(tf.int32, [None, 10], name="Y")

# prepare param
hidden_1 = 256
hidden_2 = 256
input = 784
classification = 10

weights = {
    'h1': tf.Variable(tf.random_normal([input, hidden_1]), name="w1"),
    'h2': tf.Variable(tf.random_normal([hidden_1, hidden_2]), name="w2"),
    'out': tf.Variable(tf.random_normal([hidden_2, classification]), name="w")
}

bias = {
    'b1': tf.Variable(tf.random_normal([hidden_1]), name="b1"),
    'b2': tf.Variable(tf.random_normal([hidden_2]), name="b2"),
    'out': tf.Variable(tf.random_normal([classification]), name="bias")
}


def multi_layer_perception(x, weights, biases):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name="fc1")
    layer1 = tf.nn.relu(layer1, name="relu1")

    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'], name="fc2")
    layer2 = tf.nn.relu(layer2, name="relu2")

    out_layer = tf.add(tf.matmul(layer2, weights['out']), biases['out'], name="fc3")
    return out_layer


pred = multi_layer_perception(X, weights, bias)

learning_rate = 0.01
loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y, name='cross_entropy')
loss = tf.reduce_mean(loss_all, name="reduce_loss")

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

training_epoch = 15

batch_size = 128
display_step = 1

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("C:/Temp/log/multi", sess.graph)
    writer.close()

    n_batches = int(mnist.train.num_examples / batch_size)

    for epoch in range(training_epoch):
        avg_loss = 0

        for i in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            avg_loss += l / n_batches

        if epoch % display_step == 0:
            print('epoch:', '%04d' % (epoch + 1), 'cost = ', \
                  "{:.9f}".format(avg_loss))

    print("done")

    correct_preds = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    print("Accuracy: ", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
