import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# prepare data
obv = 100
xs = np.linspace(-3, 3, obv)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, obv)

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

Y_predict = tf.add(tf.multiply(X, w), b)

w2 = tf.Variable(tf.random_normal([1]), name="weight2")
Y_predict = tf.add(tf.multiply(tf.pow(X, 2), w2), Y_predict)
w3 = tf.Variable(tf.random_normal([1]), name="weight3")
Y_predict = tf.add(tf.multiply(tf.pow(X, 3), w3), Y_predict)

n_sample = xs.shape[0]
loss = tf.reduce_sum(tf.pow(Y_predict - Y, 2)) / n_sample

learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("C:/Temp/log/polynomial", sess.graph)
    writer.close()

    for i in range(1000):
        total_loss = 0
        for x, y in zip(xs, ys):
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        if i % 20 == 0:
            print("Epoch {0}: {1}".format(i, total_loss / n_sample))

    w, w2, w3, b = sess.run([w, w2, w3, b])

print(w, w2, w3, b)

# show
plt.rcParams["figure.figsize"] = (14, 8)
plt.scatter(xs, ys)
plt.plot(xs, ys, 'bo', label="real data")
plt.plot(xs, xs * w + np.power(xs, 2) * w2 + np.power(xs, 3) * w3 + b, 'r:', label="prediction")
plt.legend()
plt.show()
