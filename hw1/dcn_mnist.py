__author__ = 'tan_nguyen'

import os
import time

# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import Tensorflow and start a session
import tensorflow as tf
sess = tf.InteractiveSession()
RUN = '/run_ReLU_X_adam_lr1e_4'

def weight_variable(shape, name):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    # initial = tf.truncated_normal(shape, stddev=0.1)
    return initial

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''
    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''
    # IMPLEMENT YOUR CONV2D HERE
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''
    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def leaky_relu(X, alpha=0.1):
    """
    Implementation of leaky ReLU unit.
    :param X: input vector
    :param alpha: slope of the negative portion
    :return:
    """
    return tf.maximum(X, alpha * X)


def nn_layer(input_tensor, input_channel, output_channel, window, name):
    """
    Generalized conv layer with a convolution and nonlinearity
    :param input_tensor: input tensor that will be multiplied with
    :param input_channel: dimension of the input channel
    :param output_channel: dimension of the output channel
    :param window: window size
    :param name: name of layer
    :return: output of the run
    """
    add_summaries(input_tensor, name + '.input')
    w_conv = weight_variable([window, window, input_channel, output_channel], name+'.w')
    add_summaries(w_conv, name + '.w')
    b_conv = bias_variable([output_channel])
    add_summaries(b_conv, name + '.b')
    h_conv = tf.nn.relu(conv2d(input_tensor, w_conv) + b_conv)
    add_summaries(h_conv, name+'.activation')
    pooled = max_pool_2x2(h_conv)
    add_summaries(pooled, name+'.maxpool')
    return pooled

def add_summaries(tensor, name):
    """
    add the data to the summary writer
    :param tensor: tensor to analyze
    :param name: name of tensor
    :return: None
    """
    name = name
    mean = tf.reduce_mean(tensor)
    tf.scalar_summary(name + '.mean', mean)
    std = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
    tf.scalar_summary(name + '.std', std)
    tf.scalar_summary(name + '.max', tf.reduce_max(tensor))
    tf.scalar_summary(name + '.min', tf.reduce_min(tensor))
    tf.histogram_summary(name, tensor)

def get_accuracies(accuracy, x, y_, keep_prob):
    """
    get the validation and test accuracies
    :return:
    """
    test_acc = accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    val_acc = accuracy.eval(feed_dict={
        x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})

    return test_acc, val_acc

def main():
    # Specify training parameters
    result_dir = './results/' # directory where the results from the training are saved
    max_step = 5500 # the maximum iterations. After max_step iterations, the training will stop no matter what

    start_time = time.time() # start timing

    # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK
    SIZE = 28 * 28
    NUM_CLASSES = 10

    # placeholders for input data and input labeles
    x = tf.placeholder(tf.float32, shape=[None, SIZE])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    # reshape the input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    layer_1 = nn_layer(x_image, 1, 32, 5, 'layer_1')

    # second convolutional layer
    layer_2 = nn_layer(layer_1, 32, 64, 5, 'layer_2')

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024], 'wfc1')
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(layer_2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    # softmax
    W_fc2 = weight_variable([1024, 10], 'wfc2')
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING
    # global_step = tf.Variable(0, trainable=False)
    # starter_rate = 1e-1
    # learning_rate = tf.train.exponential_decay(starter_rate, global_step, 800, .98, staircase=True)
    # setup training
    cross_entropy = tf.reduce_mean(-1 * tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(cross_entropy.op.name, cross_entropy)
    sess.run(tf.initialize_all_variables())
    test_acc, val_acc = get_accuracies(accuracy, x, y_, keep_prob)
    test_sum = tf.scalar_summary('test_error', test_acc)
    val_sum = tf.scalar_summary('validation_error', val_acc)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    # init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    train_writer = tf.train.SummaryWriter(result_dir + RUN + '/train', sess.graph)
    error_writer = tf.train.SummaryWriter(result_dir + RUN + '/error')
    # Run the Op to initialize the variables.

    for i in range(max_step):
        batch = mnist.train.next_batch(50) # make the data batch, which is used in the training iteration.

        # Define here to capture scoping
        def feed_dict(test=False, validation=False):
            """
            assign the proper params to the feed dict give that we're looking for test or validation
            :param test:
            :return:
            """
            if test:
                return {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
            elif validation:
                return {x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}
            else:
                return {x: batch[0], y_: batch[1], keep_prob: 0.5}

        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

            test_acc, val_acc = get_accuracies(accuracy, x, y_, keep_prob)
            print("test accuracy %g" % test_acc)
            print("validation accuracy %g"% val_acc)
            summary_str = sess.run(summary_op,feed_dict=feed_dict())
            train_writer.add_summary(summary_str, i)
            train_writer.flush()


        elif i%100 == 0:

            # output the training accuracy every 100 iterations
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_:batch[1], keep_prob: 1.0})
            # print("step %d, training accuracy %g"%(i, train_accuracy))

            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict=feed_dict())
            train_writer.add_summary(summary_str, i)
            train_writer.flush()

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # run one train_step

    # print test error
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    stop_time = time.time()
    print('The training takes %f second to finish'%(stop_time - start_time))

if __name__ == "__main__":
    main()
