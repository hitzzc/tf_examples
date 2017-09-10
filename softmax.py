import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/MNIST_data/", one_hot=True)

# define variable
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define input, output placeholder
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# compute y
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss, using log entropy loss
L = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(L)

# create session and initialize variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
	i, o = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: i, y_: o})

# test
pred = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(pred)
print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))