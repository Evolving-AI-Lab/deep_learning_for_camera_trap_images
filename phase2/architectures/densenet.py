import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import common
import math


def inference(x, depth, num_output, dropout_rate, wd, is_training, transfer_mode= False):
    stages= []
    K=32
    if depth == 121:
      stages= [6, 12, 24, 16]
    elif depth == 169:
      stages= [6, 12, 32, 32]
    elif depth == 201:
      stages= [6, 12, 48, 32]
    elif depth == 161:
      stages= [6, 12, 36, 24]
      K= 48

    return getModel(x, num_output, K, stages, dropout_rate, wd, is_training, transfer_mode= transfer_mode)

def full_conv(x, K, is_training, dropout_rate, wd):
  print("Full Conv",x)
  with tf.variable_scope('conv1x1'):
    current = common.batchNormalization(x, is_training=is_training)
    current = tf.nn.relu(current)
    current = common.spatialConvolution(current, 1, 1, 4*K, wd=wd)
    print("Full Conv Out", current)
    #current = tf.nn.dropout(current, dropout_rate)
  with tf.variable_scope('conv3x3'):
    current = common.batchNormalization(current, is_training=is_training)
    current = tf.nn.relu(current)
    current = common.spatialConvolution(current, 3, 1, K, wd=wd)
  return current

def block(x, layers, K, is_training, dropout_rate, wd):
  current = x
  for idx in xrange(layers):
    with tf.variable_scope('L'+str(idx)):
      tmp = full_conv(current, K, is_training, dropout_rate, wd= wd)
      current = tf.concat((current, tmp),3)
  return current

def transition(x, K, dropout_rate, wd, is_training):
  with tf.variable_scope('conv'):
    current = common.batchNormalization(x, is_training=is_training)
    current = tf.nn.relu(current)
    shape = current.get_shape().as_list()
    dim = math.floor(shape[3]*0.5)
    current = common.spatialConvolution(current, 1, 1, dim, wd=wd)
    #current = tf.nn.dropout(current, dropout_rate)
    current = common.avgPool(current, 2, 2)
  return current

def getModel(x, num_output, K, stages, dropout_rate, wd, is_training, transfer_mode= False):
    print("input",x)
    with tf.variable_scope('conv1'):
        x = common.spatialConvolution(x, 7, 2, 2*K, wd= wd)
        print("First Conv",x)
        x = common.batchNormalization(x, is_training= is_training)
        x = tf.nn.relu(x)
        x = common.maxPool(x, 3, 2)
        print("First Maxpool",x)
        
    with tf.variable_scope('block1'):
        x = block(x, stages[0], K, is_training= is_training, dropout_rate= dropout_rate, wd= wd)
        print("block1",x)

    with tf.variable_scope('trans1'):
        x = transition(x, K, dropout_rate= dropout_rate, wd= wd, is_training= is_training)    
        print("transition1",x)

    with tf.variable_scope('block2'):
        x = block(x, stages[1], K, is_training= is_training, dropout_rate= dropout_rate, wd= wd)
        print("block2",x)

    with tf.variable_scope('trans2'):
        x = transition(x, K, dropout_rate= dropout_rate, wd= wd, is_training= is_training)    
        print("transition2",x)

    with tf.variable_scope('block3'):
        x = block(x, stages[2], K, is_training= is_training, dropout_rate= dropout_rate, wd= wd)
        print("block3",x)

    with tf.variable_scope('trans3'):
        x = transition(x, K, dropout_rate= dropout_rate, wd= wd, is_training= is_training)    
        print("transition3",x)

    with tf.variable_scope('block4'):
        x = block(x, stages[3], K, is_training= is_training, dropout_rate= dropout_rate, wd= wd)
        print("block4",x)

    x = common.avgPool(x,7,1, padding= 'VALID')
    print("Last Avg Pool",x)

    x= common.flatten(x)
    print("flatten",x)

    output = [None]*8
    with tf.variable_scope('output0'):
      output[0] = common.fullyConnected(x, 48, wd= wd)
    with tf.variable_scope('output1'):
      output[1] = common.fullyConnected(x, 12, wd= wd)
    for o in xrange(2,8):
      with tf.variable_scope('output'+str(o)):
        output[o] = common.fullyConnected(x, 2, wd= wd)

    return output
