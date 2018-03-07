import tensorflow as tf
import common

def inference(x, num_output, wd, dropout_rate, is_training, transfer_mode= False):
    conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)
    fc_weight_initializer = tf.truncated_normal_initializer(stddev= 0.01)
 
    with tf.variable_scope('conv1'):
      network = common.spatialConvolution(x, 11, 4, 64, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu (network)
      #common.activation_summary(network)
    network = common.maxPool(network, 3, 2)
    with tf.variable_scope('conv2'):
      network = common.spatialConvolution(network, 5, 1, 192, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu(network)
      #common.activation_summary(network)
    network = common.maxPool(network, 3, 2)
    with tf.variable_scope('conv3'):
      network = common.spatialConvolution(network, 3, 1, 384, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu(network)
      #common.activation_summary(network)
    with tf.variable_scope('conv4'):
      network = common.spatialConvolution(network, 3, 1, 256, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu(network)
    with tf.variable_scope('conv5'):
      network = common.spatialConvolution(network, 3, 1, 256, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu(network)
    network = common.maxPool(network, 3, 2)
    network = common.flatten(network)
    with tf.variable_scope('fc1'): 
      network = tf.nn.dropout(network, dropout_rate)
      network = common.fullyConnected(network, 4096, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu(network)
    with tf.variable_scope('fc2'):
      network = tf.nn.dropout(network, dropout_rate)
      network = common.fullyConnected(network, 4096, wd= wd)
      network = common.batchNormalization(network, is_training= is_training)
      network = tf.nn.relu(network)
    output = [None]*len(num_output)
    for o in xrange(0,len(num_output)):
      with tf.variable_scope('output'+str(o)):
        output[o] = common.fullyConnected(network, num_output[o], weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)

    return output
