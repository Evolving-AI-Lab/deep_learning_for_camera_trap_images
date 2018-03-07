import tensorflow as tf
import common

def inference(x, num_output, wd, dropout_rate, is_training, transfer_mode= False, model_type= 'A'):
   # Create tables describing VGG configurations A, B, D, E
   if model_type == 'A':
      config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
   elif model_type == 'B':
      config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
   elif model_type == 'D':
      config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
   elif model_type == 'E':
      config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
   else:
      print('Unknown model type: ' + model_type + ' | Please specify a modelType A or B or D or E')

   network= x 
   
   for i,v in enumerate(config):
     if v == 'M':
       network= common.maxPool(network, 2, 2)
     else:  
       with tf.variable_scope('conv'+str(i)):
         network = common.spatialConvolution(network, 3, 1, v, wd= wd)
         network = common.batchNormalization(network, is_training= is_training)
         network = tf.nn.relu(network)

   network= common.flatten(network)

   with tf.variable_scope('fc1'): 
     network = common.fullyConnected(network, 4096, wd= wd)
     network = common.batchNormalization(network, is_training= is_training)
     network = tf.nn.relu(network)
     network = tf.nn.dropout(network, dropout_rate)
   with tf.variable_scope('fc2'):
     network = common.fullyConnected(network, 4096, wd= wd)
     network = common.batchNormalization(network, is_training= is_training)
     network = tf.nn.relu(network)
     #network = tf.nn.dropout(network, dropout_rate)
   output = [None]*len(num_output)
   for o in xrange(0,len(num_output)):
     with tf.variable_scope('output'+str(o)):
       output[o] = common.fullyConnected(network, num_output[o],  wd= wd)

   return output
