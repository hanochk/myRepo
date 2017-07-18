
import numpy as np
from tensorflow_confusion_metrics import tf_confusion_metrics
import matplotlib.pyplot as plt
from Var_summ import variable_summaries
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from plot_conf_mat import plot_confusion_matrix


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape: object) -> object:
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])

#input image is 28x28 =>1st Conv same size + maxpooling=>14x14x32
#1st convolution layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#2nd convolution layer
#input : 14x14x32 = > Conv maintain the same size => max-pooling => 7x7x64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#Fully connected
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
#To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.1
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#Train and evaluate
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


#Tensor board initialization
variable_summaries(W_conv2, "W_conv2")
variable_summaries(b_conv2, "b_conv2")
tf.scalar_summary("cross_entropy:", cross_entropy)
summary_op = tf.merge_all_summaries()
summary_writer = tf.summary.FileWriter("mnist_CNN_tf_log", graph=sess.graph)

# validLen = 500;
# validImgs = mnist.validation.images[:validLen ,:] #GTX can't handle amount of memory
# validLbls = mnist.validation.labels[:validLen ,:]
for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
#    train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
    [accuracyO, summary_str] = sess.run([accuracy, summary_op], feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, accuracyO))
  if i%100 == 10:
    X_batch, Y_batch = mnist.validation.next_batch(512)
    [accuracyO,cc,summary_str] = (sess.run([accuracy,cross_entropy,summary_op], feed_dict={x: X_batch, y_: Y_batch, keep_prob: 1.0}))
    print("step %d, validation accuracy %g"%(i, accuracyO))


  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

