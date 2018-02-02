import tensorflow as tf


class Predict(object):

    save_name = 'trained_model'

    def __init__(self):
        self.saver = tf.train.import_meta_graph('{name}.meta'.format(name=self.save_name))
