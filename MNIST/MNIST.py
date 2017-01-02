import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.01
training_iter = 30
batch_size    = 100
display_step  = 2

# tensorflow graph model
x = tf.placeholder("float", [None, 784])  #mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10])   # 0-9 digits hence 10 classes

# Craeting model
# Model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
	model = tf.nn.softmax(tf.matmul(x,W) + b)

# data collection
w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("biases", b)

# Adding another scope to clean up graph representation
with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function
    tf.scalar_summary("cost_function", cost_function)

with tf.name_scope("train") as scope:
	# Gradient decent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initialize variables:
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merge_summary_op = tf.merge_all_summaries()

# Visualization :
with tf.Session() as sess:
	sess.run(init)

	# Set the logs to a folder
	summary_writer = tf.train.SummaryWriter('/Users/Hossein/MyRepos/MachineLearning/TestAndExp/TF/MNIST/logs', graph_def=sess.graph_def)

	# Training time:
	for iter in range(training_iter):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(optimizer, feed_dict={x: batch_xs, y:batch_ys})
			avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
			summary_str = sess.run(merge_summary_op, feed_dict={x: batch_xs, y:batch_ys})
			summary_writer.add_summary(summary_str, iter*total_batch+i)
		if iter%display_step == 0:
			print "Iteration: ", '%04d'%(iter+1), "cost=", "{:.9f}".format(avg_cost)

	print "Comleted Tuning"

	predict  = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(predict, "float"))
	print "Accuracy: ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}) 













