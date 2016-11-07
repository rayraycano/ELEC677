from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp

# --------------------------------------------------
# setup

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
    W = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return W

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
    h_conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1,2,2,1],
                           strides=[1,2,2,1], padding='SAME')
    return h_max


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


ntrain = 1000# per class
ntest = 100# per class
nclass = 10# number of classes
imsize = 28
nchannels = 3
batchsize = 100

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

# Load images from folders. 
itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = '~/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = '~/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

#tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_data = tf.placeholder(shape=[None, imsize, imsize, nchannels])

#tf variable for labels
tf_labels = tf.placeholder(shape=[None, nclass])

# --------------------------------------------------
# model
#create your model
layer1 = nn_layer(tf_data, 3, 32, 5, 'layer1')
layer2 = nn_layer(layer1, 32, 64, 5, 'layer2')
W_fc1 = weight_variable([7 * 7 * 64, 1024], 'wfc1')
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(layer2, [-1, 7*7*64])
h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
W_fc2 = weight_variable([1024, 10], 'wfc2')
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(h_fc2)

# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy

global_step = tf.Variable(0, trainable=False)
starter_rate = 1e-4
learning_rate = tf.train.exponential_decay(starter_rate, global_step, 1000, .96, staircase=True)

cross_entropy = tf.reduce_mean(-1 * tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
# optimization
tf.scalar_summary(cross_entropy.op.name, cross_entropy)
sess.run(tf.initialize_all_variables())

sess.run(tf.initialize_all_variables())

#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_xs = [batchsize, imsize, imsize, nchannels]
batch_ys = [batchsize, nchannels]#setup as [batchsize, the how many classes]

for i in range(1000): # try a small iteration size once it works then continue
    perm = np.arange(ntrain)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%10 == 0:
        # calculate train accuracy and print it
        train_accuracy = accuracy.eval(feed_dict={
            tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

    # dropout only during training
    optimizer.run(feed_dict={tf_data:batch_xs, tf_labels:batch_ys,
                             keep_prob:0.5})

# --------------------------------------------------
# test

print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))


sess.close()
