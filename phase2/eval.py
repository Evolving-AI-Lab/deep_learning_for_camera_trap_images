"""Evaluating a trained model on the test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf
import argparse
import arch
import data_loader
import sys


def evaluate(args):

  # Building the graph
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    # Get images and labels.
    images, labels, urls = data_loader.read_inputs(False, args)
    # Performing computations on a GPU
    with tf.device('/gpu:0'):
      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = arch.get_model(images, 0.0, False, args)

      # Calculate predictions accuracies top-1 and top-5
      top1acc= [None]*len(logits)
      for i in xrange(0,len(logits)):
        top1acc[i] = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits[i], labels[:,i], 1), tf.float32))
      # Top-5 ID accuracy
      top5acc_id= tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits[0], labels[:,0], 5), tf.float32))
      # Top-3 count accuracy
      top3acc_cn= tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits[1], labels[:,1], 3), tf.float32)) 
      # The percent of predictions within +/1 bin 
      one_bin_off_loss = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(tf.cast(tf.argmax(logits[1], axis=1), tf.float64)-tf.cast(labels[:,1],tf.float64)),1),tf.float32))
        
      # Information about the predictions for saving in a file
        
      # Species Identification
      top5_id = tf.nn.top_k(tf.nn.softmax(logits[0]), 5)
      top5ind_id= top5_id.indices
      top5val_id= top5_id.values
      # Count
      top3_cn = tf.nn.top_k(tf.nn.softmax(logits[1]), 3)
      top3ind_cn= top3_cn.indices
      top3val_cn= top3_cn.values
      # Additional Attributes (e.g. description)
      top1_bh= [None]*6
      top1ind_bh= [None]*6
      top1val_bh= [None]*6

      for i in xrange(0,6):
        top1_bh[i]= tf.nn.top_k(tf.nn.softmax(logits[i+2]), 1)
        top1ind_bh[i]= top1_bh[i].indices
        top1val_bh[i]= top1_bh[i].values

      # Binarizing the additial attributes predictions
      binary_behavior_logits= tf.cast([top1ind_bh[0],top1ind_bh[1],top1ind_bh[2],top1ind_bh[3],top1ind_bh[4], top1ind_bh[5]],tf.int32)
      # Cast to Boolean
      binary_behavior_predictions = tf.squeeze(tf.cast(binary_behavior_logits,tf.bool))
      # Group labels together
      binary_behavior_labels_logits = [labels[:,2],labels[:,3],labels[:,4],labels[:,5],labels[:,6], labels[:,7]]
      # Cast labels to Boolean
      binary_behavior_labels = tf.cast(binary_behavior_labels_logits,tf.bool)

      # Compute the size of label sets (for each image separately)
      y_length = tf.reduce_sum(binary_behavior_labels_logits,axis=0) 
      # Compute the size of prediction sets (for each image separately)
      z_length = tf.reduce_sum(binary_behavior_logits,axis=0)
      # Compute the union of the labels set and prediction set
      union_length= tf.reduce_sum(tf.cast(tf.logical_or(binary_behavior_labels,binary_behavior_predictions),tf.int32),axis=0)
      # Compute the intersection of the labels set and prediction set
      intersect_length= tf.reduce_sum(tf.cast(tf.logical_and(binary_behavior_labels,binary_behavior_predictions),tf.int32),axis=0)

      # For reading the snapshot files from file
      saver = tf.train.Saver(tf.global_variables())

      # Build the summary operation based on the TF collection of Summaries.
      summary_op = tf.summary.merge_all()

      summary_writer = tf.summary.FileWriter(args.log_dir, g)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      ckpt = tf.train.get_checkpoint_state(args.log_dir)

      # Load the latest model
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)

      else:
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      true_predictions_count = [0]*8  # Counts the number of correct top-1 predictions
      true_top5_predictions_count = 0 # Counts the number of correct top-5 predictions for species identification
      true_top3_predictions_count = 0 # Counts the number of correct top-3 predictions for counting

      accv_all = 0 # Counts accuracy of additional attributes
      prcv_all = 0 # Counts precision of additional attributes
      recv_all = 0 # Counts recall of additional attributes
      total_examples= 0 # Counts number of total examples
 
      one_bin_off_val = 0
      step = 0
      predictions_format_str = ('%d,%s,%s,%s,%s,%s,%s,%s,%s]\n')
      batch_format_str = ('Batch Number: %d, Top-1 Accuracy: %s, Top-5 Accuracy: %.3f, Top-3 Accuracy: %.3f, One bin off Loss: %.3f, Accuracy: %.3f, Precision: %.3f, Recall: %.3f')

      # Output file to save predictions and their confidences
      out_file = open(args.save_predictions,'w')

      while step < args.num_batches and not coord.should_stop():
 
        top1_accuracy, top5_accuracy, top3_accuracy, urls_values, label_values, top5guesses_id, top5conf, top3guesses_cn, top3conf, top1guesses_bh, top1conf, obol_val, yval, zval, uval, ival = sess.run([top1acc, top5acc_id, top3acc_cn, urls, labels, top5ind_id, top5val_id, top3ind_cn, top3val_cn, top1ind_bh, top1val_bh, one_bin_off_loss, y_length, z_length, union_length, intersect_length])
        for i in xrange(0,urls_values.shape[0]):
          out_file.write(predictions_format_str%(step*args.batch_size+i+1, urls_values[i],
              '[' + ', '.join('%d' % np.asscalar(item) for item in label_values[i]) + ']',
              '[' + ', '.join('%d' % np.asscalar(item) for item in top5guesses_id[i]) + ']',
              '[' + ', '.join('%.3f' % np.asscalar(item) for item in top5conf[i]) + ']',
              '[' + ', '.join('%d' % np.asscalar(item) for item in top3guesses_cn[i]) + ']',
              '[' + ', '.join('%.3f' % np.asscalar(item) for item in top3conf[i]) + ']',
              '[' + ', '.join('%d' % np.asscalar(item) for item in [top1guesses_bh[0][i],top1guesses_bh[1][i],top1guesses_bh[2][i],top1guesses_bh[3][i],top1guesses_bh[4][i],top1guesses_bh[5][i]]) + ']',
              '[' + ', '.join('%.3f' % np.asscalar(item) for item in [top1conf[0][i],top1conf[1][i],top1conf[2][i],top1conf[3][i],top1conf[4][i],top1conf[5][i]]) + ']'))
          out_file.flush()
        total_examples+= uval.shape[0]

        # Computing Accuracy, Precision, and Recall of additional attributes
        for i in xrange(0,uval.shape[0]):
          if(uval[i]==0):
            # If both the label set and prediction set are empty, it is a correct prediction
            accv_all+= 1
          else:
            accv_all+= ival[i]/uval[i]
          if(np.asscalar(yval[i])==0):
            # If the lebal set is empty, then recall is 100%
            recv_all+= 1
          else:
            recv_all+= np.asscalar(ival[i])/yval[i]
          if(zval[i]==0):
            # if The prediction set is empty then precision is 100%
            prcv_all+= 1
          else:
            prcv_all+= ival[i]/zval[i]

        for i in xrange(0,len(logits)):
          true_predictions_count[i] += top1_accuracy[i]

        true_top5_predictions_count+= top5_accuracy
        true_top3_predictions_count+= top3_accuracy
        one_bin_off_val+= obol_val

        print(batch_format_str%(step, '[' + ', '.join('%.3f' % (item/(step+1.0)) for item in true_predictions_count) + ']', true_top5_predictions_count/(step + 1.0), true_top3_predictions_count/(step+1.0), obol_val/(step+1.0), accv_all/total_examples, prcv_all/total_examples, recv_all/total_examples))
        sys.stdout.flush()
        step += 1

      out_file.close()
 
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      coord.request_stop()
      coord.join(threads)

def main():
  parser = argparse.ArgumentParser(description='Process Command-line Arguments')
  parser.add_argument('--load_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images for loading from disk')
  parser.add_argument('--crop_size', nargs= 2, default= [224,224], type= int, action= 'store', help= 'The width and height of images after random cropping')
  parser.add_argument('--batch_size', default= 512, type= int, action= 'store', help= 'The testing batch size')
  parser.add_argument('--num_classes', default= [48, 12, 2, 2, 2, 2, 2, 2] , type=int, nargs= '+', help= 'The number of classes')
  parser.add_argument('--num_channels', default= 3, type= int, action= 'store', help= 'The number of channels in input images')
  parser.add_argument('--num_batches' , default=-1 , type= int, action= 'store', help= 'The number of batches of data')
  parser.add_argument('--path_prefix' , default='/project/EvolvingAI/mnorouzz/Serengiti/resized', action= 'store', help= 'The prefix address for images')
  parser.add_argument('--delimiter' , default=',', action = 'store', help= 'Delimiter for the input files')
  parser.add_argument('--data_info'   , default= 'gold_expert_info.csv', action= 'store', help= 'File containing the addresses and labels of testing images')
  parser.add_argument('--num_threads', default= 20, type= int, action= 'store', help= 'The number of threads for loading data')
  parser.add_argument('--architecture', default= 'resnet', help='The DNN architecture')
  parser.add_argument('--depth', default= 50, type= int, help= 'The depth of ResNet architecture')
  parser.add_argument('--log_dir', default= None, action= 'store', help='Path for saving Tensorboard info and checkpoints')
  parser.add_argument('save_predictions', default= None, action= 'store', help= 'Save predictions of the networks along with their confidence in the specified file')

  args = parser.parse_args()
  args.num_samples = sum(1 for line in open(args.data_info))
  if args.num_batches==-1:
    if(args.num_samples%args.batch_size==0):
      args.num_batches= int(args.num_samples/args.batch_size)
    else:
      args.num_batches= int(args.num_samples/args.batch_size)+1

  print(args)

  evaluate(args)


if __name__ == '__main__':
  main()
