import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import keras.backend
import numpy as np
import random
import keras.models
import keras.layers

disable_eager_execution()


class DQNet:
    def __init__(self, state_size, action_size, learning_rate, name='DQNet'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(name_or_scope='inputs'):
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.action_ = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name='actions_')

            self.target_q = tf.compat.v1.placeholder(tf.float32, [None], name='target')

            self.conv1 = tf.compat.v1.layers.Conv2D(inputs=self.inputs_,
                                                    filters=32,
                                                    kernel_size=[8, 8],
                                                    strides=[4, 4],
                                                    padding='VALID',
                                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                                    name='conv1')

            self.conv1_out = tf.compat.v1.nn.elu(self.conv1, name='conv1_out')

            self.conv2 = tf.compat.v1.layers.conv2d(input=self.inputs_,
                                                    filters=64,
                                                    kernel_size=[4, 4],
                                                    strides=[2, 2],
                                                    padding='VALID',
                                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                                    name='conv2')

            self.conv2_out = tf.compat.v1.nn.elu(self.conv2, name='conv2_out')

            self.conv3 = tf.compat.v1.layers.conv2d(input=self.inputs_,
                                                    filters=32,
                                                    kernel_size=[3, 3],
                                                    strides=[2, 2],
                                                    padding='VALID',
                                                    kernel_initializer=tf.keras.initializers.GlorotNormal,
                                                    name='conv3')

            self.conv3_out = tf.compat.v1.nn.elu(self.conv3, name='conv3_out')

            self.flatten = tf.compat.v1.layers.Flatten(self.conv3_out)

            self.fc = tf.compat.v1.layers.Dense(input=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.keras.initializers.GlorotNormal,
                                                name='fc1')

            self.output = tf.compat.v1.layers.Dense(input=self.fc,
                                                    units=self.action_size,
                                                    activation=None,
                                                    kernel_initializer=tf.keras.initializers.GlorotNormal)

            self.Q = tf.compat.v1.reduce_sum(tf.multiply(self.output, self.action_))

            self.loss = tf.compat.v1.reduce_mean(tf.square(self.target_q - self.Q))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
