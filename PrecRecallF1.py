#http://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
pred = multilayer_perceptron(x, weights, biases)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
init = tf.initialize_all_variables()
sess.run(init)
for epoch in xrange(150):
		for i in xrange(total_batch):
				train_step.run(feed_dict = {x: train_arrays, y: train_labels})
				avg_cost += sess.run(cost, feed_dict={x: train_arrays, y: train_labels})/total_batch         
		if epoch % display_step == 0:
				print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

#metrics
y_p = tf.argmax(pred, 1)
val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_arrays, y:test_label})

print "validation accuracy:", val_accuracy
y_true = np.argmax(test_label,1)
print "Precision", sk.metrics.precision_score(y_true, y_pred)
print "Recall", sk.metrics.recall_score(y_true, y_pred)
print "f1_score", sk.metrics.f1_score(y_true, y_pred)
print "confusion_matrix"
print sk.metrics.confusion_matrix(y_true, y_pred)
fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)