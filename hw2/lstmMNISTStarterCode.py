import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data

#call mnist function
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

result_dir = './rnn_results/'
RUN = 'rnn_lrED_128_3e5'
CELL_TYPE = 'rnn'

global_step = tf.Variable(0, trainable=False)
learningRate = tf.train.exponential_decay(.001, global_step=global_step, decay_rate=.97, decay_steps=10000)
# learningRate = .001
trainingIters = 3e5
batchSize = 128
displayStep = 40

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 128 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

x_ = tf.placeholder('float', [None, nSteps, nInput])
y_truth = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(0, nSteps, x) #configuring so you can get it as needed for the 28 pixels

    if CELL_TYPE == 'rnn':
        cell = tf.nn.rnn_cell.BasicRNNCell(nHidden)
    elif CELL_TYPE == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(nHidden)
    else:
        cell = tf.nn.rnn_cell.GRUCell(nHidden)
    outputs, state = tf.nn.rnn(cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out'] + biases['out'])

pred = RNN(x_, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.nn.softmax_cross_entropy_with_logits(pred, y_truth, name='softmax_w_logits')
cross_entropy = tf.reduce_mean(-1 * tf.reduce_sum(y_truth * tf.log(pred), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost, global_step=global_step)


correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary(cross_entropy.op.name, cross_entropy)
summary_op = tf.merge_all_summaries()
init = tf.initialize_all_variables()

train_file = open(RUN + '_train.txt', 'w')
index_file = open(RUN + '_index.txt', 'w')
val_file = open(RUN + "_test.txt", 'w')
with tf.Session() as sess:
    sess.run(init)
    step = 1
    sum_writer = tf.train.SummaryWriter(result_dir + RUN, sess.graph)

    while step * batchSize < trainingIters:
        batchX, batchY = mnist.train.next_batch(batch_size=batchSize)#mnist has a way to get the next batch
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        summary_string = sess.run(summary_op, feed_dict={x_: batchX, y_truth: batchY})
        sum_writer.add_summary(summary_string)
        sum_writer.flush()

        if step % displayStep == 0:
            valX = mnist.validation.images
            valX = valX.reshape([-1, nSteps, nInput])

            train_cost = cost.eval(session=sess, feed_dict={x_: batchX, y_truth: batchY})

            val_cost = cost.eval(session=sess, feed_dict={x_: valX, y_truth: mnist.validation.labels})
            print("train cost at " + str(step*batchSize) + "= " + str(np.mean(train_cost)))
            print("val cost: " + str(np.mean(val_cost)))
            train_file.write(str(np.mean(train_cost)) + ',')
            val_file.write(str(np.mean(val_cost)) + ',')
            index_file.write(str(step*batchSize) + ',')
            # print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
            #       "{:.6f}".format() + ", Training Accuracy= " + \
            #       "{:.5f}".format())
        optimizer.run(feed_dict={x_: batchX, y_truth: batchY})
        step +=1
    print('Optimization finished')

    train_file.close()
    index_file.close()
    val_file.close()

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x_: testData, y_truth:testLabel}))
