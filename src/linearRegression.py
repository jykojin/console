import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#prepare data
obv = 100
xs = np.linspace(-3, 3, obv)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, obv)

#set
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

Y_predict = tf.add(tf.multiply(X, w), b)
loss = tf.square(Y - Y_predict, name="loss")

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

n_sample = xs.shape[0]
init = tf.global_variables_initializer()

#start training
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("C:/Temp/log/linear", sess.graph)
    writer.close()

    for i in range(100):
        total_loss = 0
        for x, y in zip(xs, ys):
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
        total_loss += l

    w, b = sess.run([w, b])

print(w, b)

#show
plt.rcParams["figure.figsize"] = (14, 8)
plt.scatter(xs, ys)
plt.plot(xs, ys, 'bo', label="real data")
plt.plot(xs, xs * w + b, 'r:', label="prediction")
plt.legend()
plt.show()
