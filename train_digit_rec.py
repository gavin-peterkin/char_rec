from tensorflow.examples.tutorials.mnist import input_data

import os
import sys
import tensorflow as tf


class ConvModel(object):

    default_stddev = 0.1
    default_uniform = 0.1
    learning_rate = 1e-4

    sample_depth = 5000
    batch_size = 50
    dropout = 0.5

    save_name = 'trained_model'

    def __init__(self, load=False):

        self.sess = tf.Session()

        if not load:
            self.mnist = self.get_data()

            self._init_vars()  # Creates x, y
            self.construct_layers()
            # Initialize saver
            self.saver = tf.train.Saver()
        else:
            # Restore
            self.saver = tf.train.import_meta_graph('{name}.meta'.format(name=self.save_name))
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))

    def save(self):
        print "Saving model"
        self.saver.save(self.sess, self.save_name)

    def get_data(self):
        print "Reading data"
        return input_data.read_data_sets('MNIST_data', one_hot=True)

    def _init_vars(self):
        """Init session global vars"""
        print "Initializing session"
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

    def _get_weight_var(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=self.default_stddev))

    def _get_bias_var(self, shape):
        return tf.Variable(tf.constant(self.default_uniform, shape=shape))

    def _conv2d(self, x, W):
        """Stride is 1 zero padded"""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

    def construct_layers(self):
        print "Constructing graph"
        # Layer 1
        self.W_conv1 = self._get_weight_var([5, 5, 1, 32])
        self.b_conv1 = self._get_bias_var([32])

        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        self.h_conv1 = tf.nn.relu(self._conv2d(x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self._max_pool_2x2(self.h_conv1)
        # Layer 2
        self.W_conv2 = self._get_weight_var([5, 5, 32, 64])
        self.b_conv2 = self._get_bias_var([64])

        self.h_conv2 = tf.nn.relu(self._conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self._max_pool_2x2(self.h_conv2)
        # Fully connected layer, size reduced to 7x7
        self.W_fc1 = self._get_weight_var([7 * 7 * 64, 1024])
        self.b_fc1 = self._get_bias_var([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        # Dropout
        self.keep_prob = tf.placeholder(tf.float32)  # Prob to use dropout 0 means None
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        # Output
        self.W_fc2 = self._get_weight_var([1024, 10])
        self.b_fc2 = self._get_bias_var([10])

        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

    def train(self):
        print "Starting training..."

        x_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_conv)
        )
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(x_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_conv, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())
        for i in range(self.sample_depth):
            batch = self.mnist.train.next_batch(self.batch_size)
            if i % 100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={
                    self.x: batch[0], self.y: batch[1], self.keep_prob: self.dropout
                }, session=self.sess)
            print 'Step {:d}, training accuracy {:.3f}'.format(i, train_accuracy)
            train_step.run(feed_dict={
                self.x: batch[0], self.y: batch[1], self.keep_prob: 1.0
            }, session=self.sess)

    def test(self):
        print "Test accuracy:", self.accuracy.eval({
            self.x: self.mnist.test.images, self.y: self.mnist.test.labels,
            self.keep_prob: 1.0
        }, session=self.sess)



if __name__ == '__main__':

    command = sys.argv[1]

    if command == "save":
        model = ConvModel()
        model.train()
        model.test()
        model.save()
    else if command == "load":
        model = ConvModel(load=True)
