import tensorflow as tf

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")

c = tf.constant([2, 2], name="c")
d = tf.constant([[0, 1], [2, 3]], name="d")

x = tf.add(a, b, "simple_add")
y = tf.multiply(c, d, "VM")

h = tf.Variable(12, name="v1")

i = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('C:/Users/jinjo02/PycharmProjects/console/tflog', sess.graph)
    print(sess.run([x, y, i]))
    print(h.eval())
    writer.close()

