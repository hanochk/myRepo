import matplotlib.pyplot as plt
import numpy as np
from tensorflow_confusion_metrics import tf_confusion_metrics
from Var_summ import variable_summaries
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from plot_conf_mat import plot_confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#b = mnist.train.images[0,:10]
a = mnist.train.images
type(a)

a.size
a.shape
a.ndim
print('Tensor size', len(a))
label0 = mnist.train.labels[0]
print('Tensor label', label0)

x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

#help(tf.nn.softmax_cross_entropy_with_logits)
#Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over all the examples in the batch.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b,y_)
train_step = tf.train.GradientDescentOptimizer(0.35).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#Tensor board initialization
variable_summaries(W, "W_fc1")
variable_summaries(b, "b_fc1")
tf.scalar_summary("cross_entropy:", cross_entropy)
summary_op = tf.merge_all_summaries()
summary_writer = tf.summary.FileWriter("mnist_tf_log", graph=sess.graph)




for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  #[_, loss_cur] = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
  [_, loss_cur,summary_str] = sess.run([train_step, cross_entropy, summary_op], feed_dict={x: batch_xs, y_: batch_ys})
# sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if i%50 ==0:
      summary_writer.add_summary(summary_str, i)
      summary_writer.flush()

  if i%100 == 0:
      print ('Step %d: loss = %.2f ' % (i, loss_cur))
      print('cross_entropy = ', cross_entropy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
      # # summary_writer.add_summary(summary_str, i)
      # summary_writer.flush()

# Evaluating Our Model  How well does our model do? Well, first let's figure out where we predicted the correct label.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
errors = tf.not_equal(tf.argmax(y,1), tf.argmax(y_,1));
out = tf.argmax(y,1);
#Evaluation over the test set
[accr, predCorr, Hyp, Gt, err, pred] = sess.run([accuracy,correct_prediction,y,y_,errors,out], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#Finally, we can evaluate our accuracy on the test data. This should be about 92% correct.
print('Test accuracy',accr)


#Statistics over the test set
feed_dict_all = {x: mnist.test.images, y_:mnist.test.labels}

#for only one batch or epoch
#pred = correct_prediction.eval(feed_dict_all,sess)
#eval run another session extra under the hood
#Hyp = y.eval(feed_dict_all,sess)
#Gt = y_.eval(feed_dict_all,sess)
#err = errors.eval(feed_dict_all,sess)

correct_prediction.__sizeof__()
print('Test examples = ', Hyp.shape[0])
plt.figure(2)
plt.imshow(Hyp[:200],cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.axis('tight')


#Analyze error examples
# true_indices = []
# for val in err:
#     if val:
#         true_indices.append(i)
#
#     i += 1
arr = mnist.test.images
image = np.reshape(arr[10],[28,28])
plt.figure(1)
plt.imshow(image)

labels = mnist.test.labels
tup = np.where(err==True)
fig = np.where(labels)
figGt = fig[1]

indFail = tup[0]
plt.figure(3)
for val in indFail[:10]:
   image = np.reshape(arr[val], [28, 28])
   plt.imshow(image)
   plt.title('False detection as %s' % pred[val])


hh = tf_confusion_metrics(y, y_, sess, feed_dict_all)

cnf_matrix = confusion_matrix(figGt, pred)
#plt.imshow(cnf_matrix,cmap=plt.get_cmap('gray'), vmin=0, vmax=1000)
# Plot non-normalized confusion matrix
class_names = np.array(['0','1','2','3','4','5','6','7','8','9'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

#save model
saver = tf.train.Saver()

plt.show()

#ideas
#mse        = tf.reduce_mean(tf.square(out - out_))
#train_step = tf.train.GradientDescentOptimizer(0.3).minimize(mse)