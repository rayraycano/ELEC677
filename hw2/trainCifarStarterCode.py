from scipy import misc
import numpy as np
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.image as mpimg

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

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def leaky_relu(x, alpha=.1):
    return tf.maximum(x, alpha * x)

def nn_layer(input_tensor, input_channel, output_channel, window, name, train=tf.Variable(True)):
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
    h_norm = batch_norm(conv2d(input_tensor, w_conv), output_channel, tf_train)
    add_summaries(h_norm, name+'.batch')
    h_conv = leaky_relu(h_norm + b_conv)
    add_summaries(h_conv, name+'.activation')
    pooled = max_pool_2x2(h_conv)
    add_summaries(pooled, name+'.maxpool')
    return pooled, w_conv

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
nchannels = 1
batchsize = 64

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

# Load images from folders.
itrain = -1
itest = -1
data_dir = './data/CIFAR10/'
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = data_dir + 'Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path) # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = data_dir + 'Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path) # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable
print(LTrain[8500])

def visualize_filter(name, layer):
    x_min = tf.reduce_min(layer)
    x_max = tf.reduce_max(layer)
    kernal_01 = (layer - x_min) / (x_max - x_min)

    # to tf.image_summary fromat
    kernel_transposed = tf.transpose(kernal_01, [3, 0, 1, 2])

    return tf.image_summary(name, kernal_01, max_images=64)

sess = tf.InteractiveSession()

#tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])

#tf variable for labels
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass])

tf_train = tf.placeholder(tf.bool)

# --------------------------------------------------
# model
#create your model
layer1, filterz = nn_layer(tf_data, nchannels, 32, 5, 'layer1')
layer2, _ = nn_layer(layer1, 32, 64, 5, 'layer2')
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
learning_rate = tf.train.exponential_decay(starter_rate, global_step, 250, .97, staircase=True)

cross_entropy = tf.reduce_mean(-1 * tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
# optimization
tf.scalar_summary(cross_entropy.op.name, cross_entropy)

# Try to do a tf.split() on the filter tensor rather than transposing it
splits = tf.split(3, 32, filterz)
for i in range(len(splits)):
    if len(str(i)) == 1:
        num_str = '0' + str(i)
    else:
        num_str = str(i)
    # tf.image_summary('test', splits)

    tf.image_summary('filter' + num_str, tf.transpose(splits[i], [3, 0, 1, 2]))
# img_summary = tf.image_summary('filter', splits)
summary_op = tf.merge_all_summaries()

# Add the variable initializer Op.
# init = tf.initialize_all_variables()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
result_dir = './results/'
RUN = 'AdamLReluED1eneg3BN'
# RUN = 'testSum'
train_writer = tf.train.SummaryWriter(result_dir + RUN + '/train', sess.graph)

sess.run(tf.initialize_all_variables())

sess.run(tf.initialize_all_variables())

# setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_xs = np.zeros(shape=[batchsize, imsize, imsize, nchannels])

# setup as [batchsize, the how many classes]
batch_ys = np.zeros(shape=[batchsize, nclass])

for i in range(2000): # try a small iteration size once it works then continue
    perm = np.arange(ntrain * nclass)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j] = Train[perm[j]]
        batch_ys[j] = LTrain[perm[j]]

    summary_str = sess.run(summary_op, feed_dict={tf_data: batch_xs, tf_labels:batch_ys, keep_prob:0.5, tf_train:True})
    train_writer.add_summary(summary_str, i)
    train_writer.flush()

    if i%50 == 0:
        # calculate train accuracy and print it
        train_accuracy = accuracy.eval(feed_dict={
            tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0, tf_train:False})
        print("step %d, training accuracy %f"%(i, train_accuracy))
    # if i%100 == 0:
    #     train_writer.add_summary(img_summary, i)

    # dropout only during training
    optimizer.run(feed_dict={tf_data:batch_xs, tf_labels:batch_ys,
                             keep_prob:0.5, tf_train:True})

# Adapted from https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4#.pfea2tdvf
# Visualize the images in the filter
def plot_filters(filters):
    num_filts = filters.shape[3]
    plt.figure(1, figsize=(20, 20))
    # splits = tf.split(3, 32, filters)
    for i in range(num_filts):
        plt.subplot(7, 6, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(filters[:,:,0,i], interpolation="nearest", cmap=plt.get_cmap('gray'))
    plt.savefig('figures/filters')

def plot_activations(activations, name=''):
    num_acts = activations.shape[3]
    plt.figure(1, figsize=(30,30))

    print activations[0].shape
    print activations.shape
    for i in range(num_acts):
        plt.subplot(7,6,i+1)
        plt.imshow(activations[0,:,:,i], interpolation="nearest", cmap=plt.get_cmap('gray'))

    if name:
        plt.savefig(name)
    else:
        plt.show()


def getActivations(layer, stimuli, name=''):
    units = []
    for stim in stimuli:
        unit = layer.eval(session=sess, feed_dict={tf_data: np.reshape(stim, [1, 28, 28, 1], order='F'),
                                                keep_prob: 1.0,
                                                tf_train: False}) / len(stimuli)
        units.append(unit)
    units = np.mean(np.array(units), axis=0)
    print(units.shape)
    plot_activations(units, name)

viz = True
if viz:
    eval_filters = filterz.eval(session=sess)
    plot_filters(eval_filters)

    for j in range(0, 1000, 100):
        images = []
        for i in range(j, j + 100, 10):
            images.append(Test[i])
        getActivations(layer1, images, name='figures/activations' + RUN + str(j / 100))
    # getActivations(layer2, img)



# --------------------------------------------------
# test

print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0, tf_train:False}))


sess.close()
