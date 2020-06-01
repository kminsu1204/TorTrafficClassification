import argparse
import os

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam

#from DatasetProcessor import Dataset
#from DatasetProcessor_Scenario_B import Dataset
from DatasetProcessor_Sample_Adjustable import Dataset
#from testdataset import Dataset
from NetworkModel_conv1d import Model

def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""
    # initialize and launch graph
    with tf.Session() as sess:
        ds = Dataset(hparams.data_files_path, hparams.sampling_size, 0.2)  # sampling size must be smaller than 190 now (minimum size is 190)
        m1 = Model(sess, "m1", hparams.learning_rate)

        tf_train_dataset = tf.data.Dataset.from_tensor_slices((ds.x_train, ds.y_train)).shuffle(buffer_size=ds.train_length).batch(hparams.train_batch_size)
        
        iter = tf_train_dataset.make_one_shot_iterator()
        iter_train_dataset = iter.get_next()
        train_init_op = iter.make_initializer(tf_train_dataset)

        writer = tf.summary.FileWriter(hparams.job_dir)
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if (ds.train_length % hparams.train_batch_size == 0):
            total_batch = ds.train_length//hparams.train_batch_size
        else:
            total_batch = ds.train_length//hparams.train_batch_size + 1

        # train my model
        print('Learning started. It takes sometime...')
        summary_step = 0
        for epoch in range(hparams.num_epochs):
            avg_cost = 0

            sess.run(train_init_op)
            
            for i in range(total_batch):
                batch_xs, batch_ys = sess.run(iter_train_dataset)
                if (i % (total_batch-1) == 0):
                    c, _, s = m1.train_summary(batch_xs, batch_ys, 0.7)
                    writer.add_summary(s, global_step = summary_step)
                    summary_step += 1
                else:
                    c, _ = m1.train(batch_xs, batch_ys, 0.7)
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')

        # Test model and check accuracy
        nonTor_Audio_accuracy, nonTor_Audio_precision, nonTor_Audio_recall, nonTor_Audio_tp, nonTor_Audio_fp, nonTor_Audio_fn, nonTor_Audio_accuracy_op = m1.get_accuracy_precision_recall, nonTor_Audio(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for nonTor Audio traffic: ', nonTor_Audio_accuracy)
        print('Precision for nonTor Audio traffic: ', nonTor_Audio_precision)
        print('Recall nonTor_Audio for nonTor Audio traffic: ', nonTor_Audio_recall)
        print('True Positives for nonTor Audio traffic: ', nonTor_Audio_tp)
        #print('True Negatives for nonTor Audio traffic: ', nonTor_Audio_tn)
        print('False Positives for nonTor Audio traffic: ', nonTor_Audio_fp)
        print('False Negatives for nonTor Audio traffic: ', nonTor_Audio_fn)
        print('Accuracy based on TFPN for nonTor Audio traffic: ', nonTor_Audio_accuracy_op)

        nonTor_Browsing_accuracy, nonTor_Browsing_precision, nonTor_Browsing_recall, nonTor_Browsing_tp, nonTor_Browsing_fp, nonTor_Browsing_fn, nonTor_Browsing_accuracy_op = m1.get_accuracy_precision_recall, nonTor_Browsing(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for nonTor Browsing traffic: ', nonTor_Browsing_accuracy)
        print('Precision for nonTor Browsing traffic: ', nonTor_Browsing_precision)
        print('Recall nonTor_Browsing for nonTor Browsing traffic: ', nonTor_Browsing_recall)
        print('True Positives for nonTor Browsing traffic: ', nonTor_Browsing_tp)
        #print('True Negatives for nonTor Browsing traffic: ', nonTor_Browsing_tn)
        print('False Positives for nonTor Browsing traffic: ', nonTor_Browsing_fp)
        print('False Negatives for nonTor Browsing traffic: ', nonTor_Browsing_fn)
        print('Accuracy based on TFPN for nonTor Browsing traffic: ', nonTor_Browsing_accuracy_op)

        nonTor_Chat_accuracy, nonTor_Chat_precision, nonTor_Chat_recall, nonTor_Chat_tp, nonTor_Chat_fp, nonTor_Chat_fn, nonTor_Chat_accuracy_op = m1.get_accuracy_precision_recnonTor_Chat(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for nonTor Chat traffic: ', nonTor_Chat_accuracy)
        print('Precision for nonTor Chat traffic: ', nonTor_Chat_precision)
        print('Recall nonTor_Chat for nonTor Chat traffic: ', nonTor_Chat_recall)
        print('True Positives for nonTor Chat traffic: ', nonTor_Chat_tp)
        #print('True Negatives for nonTor Chat traffic: ', nonTor_Chat_tn)
        print('False Positives for nonTor Chat traffic: ', nonTor_Chat_fp)
        print('False Negatives for nonTor Chat traffic: ', nonTor_Chat_fn)
        print('Accuracy based on TFPN for nonTor Chat traffic: ', nonTor_Chat_accuracy_op)

        nonTor_File_accuracy, nonTor_File_precision, nonTor_File_recall, nonTor_File_tp, nonTor_File_fp, nonTor_File_fn, nonTor_File_accuracy_op = m1.get_accuracy_precision_recnonTor_File(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for nonTor File traffic: ', nonTor_File_accuracy)
        print('Precision for nonTor File traffic: ', nonTor_File_precision)
        print('Recall nonTor_File for nonTor File traffic: ', nonTor_File_recall)
        print('True Positives for nonTor File traffic: ', nonTor_File_tp)
        #print('True Negatives for nonTor File traffic: ', nonTor_File_tn)
        print('False Positives for nonTor File traffic: ', nonTor_File_fp)
        print('False Negatives for nonTor File traffic: ', nonTor_File_fn)
        print('Accuracy based on TFPN for nonTor File traffic: ', nonTor_File_accuracy_op)

        nonTor_Email_accuracy, nonTor_Email_precision, nonTor_Email_recall, nonTor_Email_tp, nonTor_Email_fp, nonTor_Email_fn, nonTor_Email_accuracy_op = m1.get_accuracy_precision_recnonTor_Email(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for nonTor Email traffic: ', nonTor_Email_accuracy)
        print('Precision for nonTor Email traffic: ', nonTor_Email_precision)
        print('Recall nonTor_Email for nonTor Email traffic: ', nonTor_Email_recall)
        print('True Positives for nonTor Email traffic: ', nonTor_Email_tp)
        #print('True Negatives for nonTor Email traffic: ', nonTor_Email_tn)
        print('False Positives for nonTor Email traffic: ', nonTor_Email_fp)
        print('False Negatives for nonTor Email traffic: ', nonTor_Email_fn)
        print('Accuracy based on TFPN for nonTor Email traffic: ', nonTor_Email_accuracy_op)

        nonTor_P2P_accuracy, nonTor_P2P_precision, nonTor_P2P_recall, nonTor_P2P_tp, nonTor_P2P_fp, nonTor_P2P_fn, nonTor_P2P_accuracy_op = m1.get_accuracy_precision_recnonTor_P2P(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for nonTor P2P traffic: ', nonTor_P2P_accuracy)
        print('Precision for nonTor P2P traffic: ', nonTor_P2P_precision)
        print('Recall nonTor_P2P for nonTor P2P traffic: ', nonTor_P2P_recall)
        print('True Positives for nonTor P2P traffic: ', nonTor_P2P_tp)
        #print('True Negatives for nonTor P2P traffic: ', nonTor_P2P_tn)
        print('False Positives for nonTor P2P traffic: ', nonTor_P2P_fp)
        print('False Negatives for nonTor P2P traffic: ', nonTor_P2P_fn)
        print('Accuracy based on TFPN for nonTor P2P traffic: ', nonTor_P2P_accuracy_op)

        nonTor_Video_accuracy, nonTor_Video_precision, nonTor_Video_recall, nonTor_Video_tp, nonTor_Video_fp, nonTor_Video_fn, nonTor_Video_accuracy_op = m1.get_accuracy_precision_recnonTor_Video(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for nonTor Video traffic: ', nonTor_Video_accuracy)
        print('Precision for nonTor Video traffic: ', nonTor_Video_precision)
        print('Recall nonTor_Video for nonTor Video traffic: ', nonTor_Video_recall)
        print('True Positives for nonTor Video traffic: ', nonTor_Video_tp)
        #print('True Negatives for nonTor Video traffic: ', nonTor_Video_tn)
        print('False Positives for nonTor Video traffic: ', nonTor_Video_fp)
        print('False Negatives for nonTor Video traffic: ', nonTor_Video_fn)
        print('Accuracy based on TFPN for nonTor Video traffic: ', nonTor_Video_accuracy_op)

        nonTor_VoIP_accuracy, nonTor_VoIP_precision, nonTor_VoIP_recall, nonTor_VoIP_tp, nonTor_VoIP_fp, nonTor_VoIP_fn, nonTor_VoIP_accuracy_op = m1.get_accuracy_precision_recnonTor_VoIP(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for nonTor VoIP traffic: ', nonTor_VoIP_accuracy)
        print('Precision for nonTor VoIP traffic: ', nonTor_VoIP_precision)
        print('Recall nonTor_VoIP for nonTor VoIP traffic: ', nonTor_VoIP_recall)
        print('True Positives for nonTor VoIP traffic: ', nonTor_VoIP_tp)
        #print('True Negatives for nonTor VoIP traffic: ', nonTor_VoIP_tn)
        print('False Positives for nonTor VoIP traffic: ', nonTor_VoIP_fp)
        print('False Negatives for nonTor VoIP traffic: ', nonTor_VoIP_fn)
        print('Accuracy based on TFPN for nonTor VoIP traffic: ', nonTor_VoIP_accuracy_op)


        tor_Audio_accuracy, tor_Audio_precision, tor_Audio_recall, tor_Audio_tp, tor_Audio_fp, tor_Audio_fn, tor_Audio_accuracy_op = m1.get_accuracy_precision_rector_Audio(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for Tor Audio traffic: ', tor_Audio_accuracy)
        print('Precision for Tor Audio traffic: ', tor_Audio_precision)
        print('Rector_Audio for Tor Audio traffic: ', tor_Audio_recall)
        print('True Positives for Tor Audio traffic: ', tor_Audio_tp)
        #print('True Negatives for Tor Audio traffic: ', tor_Audio_tn)
        print('False Positives for Tor Audio traffic: ', tor_Audio_fp)
        print('False Negatives for Tor Audio traffic: ', tor_Audio_fn)
        print('Accuracy based on TFPN for Tor Audio traffic: ', tor_Audio_accuracy_op)
       
        tor_Browsing_accuracy, tor_Browsing_precision, tor_Browsing_recall, tor_Browsing_tp, tor_Browsing_fp, tor_Browsing_fn, tor_Browsing_accuracy_op = m1.get_accuracy_precision_rector_Browsing(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for Tor Browsing traffic: ', tor_Browsing_accuracy)
        print('Precision for Tor Browsing traffic: ', tor_Browsing_precision)
        print('Rector_Browsing for Tor Browsing traffic: ', tor_Browsing_recall)
        print('True Positives for Tor Browsing traffic: ', tor_Browsing_tp)
        #print('True Negatives for Tor Browsing traffic: ', tor_Browsing_tn)
        print('False Positives for Tor Browsing traffic: ', tor_Browsing_fp)
        print('False Negatives for Tor Browsing traffic: ', tor_Browsing_fn)
        print('Accuracy based on TFPN for Tor Browsing traffic: ', tor_Browsing_accuracy_op)

        tor_Chat_accuracy, tor_Chat_precision, tor_Chat_recall, tor_Chat_tp, tor_Chat_fp, tor_Chat_fn, tor_Chat_accuracy_op = m1.get_accuracy_precision_rector_Chat(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for Tor Chat traffic: ', tor_Chat_accuracy)
        print('Precision for Tor Chat traffic: ', tor_Chat_precision)
        print('Rector_Chat for Tor Chat traffic: ', tor_Chat_recall)
        print('True Positives for Tor Chat traffic: ', tor_Chat_tp)
        #print('True Negatives for Tor Chat traffic: ', tor_Chat_tn)
        print('False Positives for Tor Chat traffic: ', tor_Chat_fp)
        print('False Negatives for Tor Chat traffic: ', tor_Chat_fn)
        print('Accuracy based on TFPN for Tor Chat traffic: ', tor_Chat_accuracy_op)
        
        tor_File_accuracy, tor_File_precision, tor_File_recall, tor_File_tp, tor_File_fp, tor_File_fn, tor_File_accuracy_op = m1.get_accuracy_precision_rector_File(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for Tor File traffic: ', tor_File_accuracy)
        print('Precision for Tor File traffic: ', tor_File_precision)
        print('Rector_File for Tor File traffic: ', tor_File_recall)
        print('True Positives for Tor File traffic: ', tor_File_tp)
        #print('True Negatives for Tor File traffic: ', tor_File_tn)
        print('False Positives for Tor File traffic: ', tor_File_fp)
        print('False Negatives for Tor File traffic: ', tor_File_fn)
        print('Accuracy based on TFPN for Tor File traffic: ', tor_File_accuracy_op)

        tor_Email_accuracy, tor_Email_precision, tor_Email_recall, tor_Email_tp, tor_Email_fp, tor_Email_fn, tor_Email_accuracy_op = m1.get_accuracy_precision_rector_Email(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for Tor Email traffic: ', tor_Email_accuracy)
        print('Precision for Tor Email traffic: ', tor_Email_precision)
        print('Rector_Email for Tor Email traffic: ', tor_Email_recall)
        print('True Positives for Tor Email traffic: ', tor_Email_tp)
        #print('True Negatives for Tor Email traffic: ', tor_Email_tn)
        print('False Positives for Tor Email traffic: ', tor_Email_fp)
        print('False Negatives for Tor Email traffic: ', tor_Email_fn)
        print('Accuracy based on TFPN for Tor Email traffic: ', tor_Email_accuracy_op)
        
        tor_P2P_accuracy, tor_P2P_precision, tor_P2P_recall, tor_P2P_tp, tor_P2P_fp, tor_P2P_fn, tor_P2P_accuracy_op = m1.get_accuracy_precision_rector_P2P(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for Tor P2P traffic: ', tor_P2P_accuracy)
        print('Precision for Tor P2P traffic: ', tor_P2P_precision)
        print('Rector_P2P for Tor P2P traffic: ', tor_P2P_recall)
        print('True Positives for Tor P2P traffic: ', tor_P2P_tp)
        #print('True Negatives for Tor P2P traffic: ', tor_P2P_tn)
        print('False Positives for Tor P2P traffic: ', tor_P2P_fp)
        print('False Negatives for Tor P2P traffic: ', tor_P2P_fn)
        print('Accuracy based on TFPN for Tor P2P traffic: ', tor_P2P_accuracy_op)

        tor_Video_accuracy, tor_Video_precision, tor_Video_recall, tor_Video_tp, tor_Video_fp, tor_Video_fn, tor_Video_accuracy_op = m1.get_accuracy_precision_rector_Video(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for Tor Video traffic: ', tor_Video_accuracy)
        print('Precision for Tor Video traffic: ', tor_Video_precision)
        print('Rector_Video for Tor Video traffic: ', tor_Video_recall)
        print('True Positives for Tor Video traffic: ', tor_Video_tp)
        #print('True Negatives for Tor Video traffic: ', tor_Video_tn)
        print('False Positives for Tor Video traffic: ', tor_Video_fp)
        print('False Negatives for Tor Video traffic: ', tor_Video_fn)
        print('Accuracy based on TFPN for Tor Video traffic: ', tor_Video_accuracy_op)

        tor_VoIP_accuracy, tor_VoIP_precision, tor_VoIP_recall, tor_VoIP_tp, tor_VoIP_fp, tor_VoIP_fn, tor_VoIP_accuracy_op = m1.get_accuracy_precision_rector_VoIP(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for Tor VoIP traffic: ', tor_VoIP_accuracy)
        print('Precision for Tor VoIP traffic: ', tor_VoIP_precision)
        print('Rector_VoIP for Tor VoIP traffic: ', tor_VoIP_recall)
        print('True Positives for Tor VoIP traffic: ', tor_VoIP_tp)
        #print('True Negatives for Tor VoIP traffic: ', tor_VoIP_tn)
        print('False Positives for Tor VoIP traffic: ', tor_VoIP_fp)
        print('False Negatives for Tor VoIP traffic: ', tor_VoIP_fn)
        print('Accuracy based on TFPN for Tor VoIP traffic: ', tor_VoIP_accuracy_op)

        #import pdb;pdb.set_trace()
        all_accuracy, all_precision, all_recall, all_tp, all_fp, all_fn, all_accuracy_op = m1.get_accuracy_precision_recall(ds.x_test, ds.y_test, 1.0)
        print('Accuracy for all traffic: ', all_accuracy)
        print('Precision for all traffic: ', all_precision)
        print('Recall for all traffic: ', all_recall)
        print('True Positives for all traffic: ', all_tp)
        #print('True Negatives for all traffic: ', all_tn)
        print('False Positives for all traffic: ', all_fp)
        print('False Negatives for all traffic: ', all_fn)
        print('Accuracy based on TFPN for all traffic: ', all_accuracy_op)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--data-files-path',
      help='GCS or local paths to dataset',
      #nargs='+',
      #required=True
  )
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int,
      default=100
  )
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=100
  )
  parser.add_argument(
      '--learning-rate',
      help='Learning rate for optimizer',
      type=float,
      default=0.0003
  )
  parser.add_argument(
      '--sampling-size',
      help='Sampling size for Dataset',
      type=int,
      default=10
  )

  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # Argument to turn on all logging
  '''parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
  )'''

  # Experiment arguments
  parser.add_argument(
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
  )
  parser.add_argument(
      '--export-format',
      help='The input format of the exported SavedModel binary',
      choices=['JSON', 'CSV', 'EXAMPLE'],
      default='CSV'
  )

  args = parser.parse_args()

  # Set python level verbosity
  #tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  #os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
  #    tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  hparams=hparam.HParams(**args.__dict__)
  run_experiment(hparams)
