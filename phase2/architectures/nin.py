import tensorflow as tf
import common

def block(x, spec, wd, is_training):
  with tf.variable_scope('conv1'):
    nin = common.spatialConvolution(x, spec[0], spec[1], spec[2], wd= wd)
    nin = common.batchNormalization(nin, is_training= is_training)
    nin = tf.nn.relu(nin)
  with tf.variable_scope('conv2'):
    nin = common.spatialConvolution(nin, 1, 1, spec[2], wd= wd)
    nin = common.batchNormalization(nin, is_training= is_training)
    nin = tf.nn.relu(nin)
  with tf.variable_scope('conv3'):
    nin = common.spatialConvolution(nin, 1, 1, spec[2], wd= wd)
    nin = common.batchNormalization(nin, is_training= is_training)
    nin = tf.nn.relu(nin)
  return nin

def inference(x, num_output, wd, is_training, transfer_mode= False):
    with tf.variable_scope('block1'):
      network = block(x, [11, 4, 96], wd, is_training)
    network = common.maxPool(network, 3, 2)
    with tf.variable_scope('block2'):
      network = block(network, [5, 1, 256], wd, is_training)
    network = common.maxPool(network, 3, 2)
    with tf.variable_scope('block3'):
      network = block(network, [3, 1, 384], wd, is_training)
    network = common.maxPool(network, 3, 2)
    with tf.variable_scope('block4'):
      network = block(network, [3, 1, 1024], wd, is_training)
    network = common.avgPool(network, 7, 1)
    network = common.flatten(network)
    output = [None]*len(num_output)
    for o in xrange(0,len(num_output)):
      with tf.variable_scope('output'+str(o)):
        output[o] = common.fullyConnected(network, num_output[o], wd= wd)

    return output
