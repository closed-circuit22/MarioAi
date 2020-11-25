# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import keras.backend
import numpy as np
import random
import keras.models
import keras.layers
from typing import Iterable

disable_eager_execution()


class DQNet:
    def __init__(
        self,
        state_size: Iterable,
        action_size: int,
        learning_rate: float,
        name: str = "DQNet",
    ):

        assert len(
            state_size) >= 3, "Conv2D takes 4 dimensional input, n x w x h x c"

        self.action_size = action_size
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(name_or_scope=name):
            self.inputs = tf.compat.v1.placeholder(
                tf.float32, [None, *self.state_size], name="inputs"
            )
            self.action = tf.compat.v1.placeholder(
                tf.float32, [None, self.action_size], name="actions"
            )
            self.target_q = tf.compat.v1.placeholder(
                tf.float32, [None], name="target")

            self.conv1 = tf.compat.v1.layers.Conv2D(
                activation="relu",
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.keras.initializers.GlorotNormal,
                name="conv1",
            )(self.inputs)

            self.conv2 = tf.compat.v1.layers.Conv2D(
                activation="relu",
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.keras.initializers.GlorotNormal,
                name="conv2",
            )(self.conv1)

            self.conv3 = tf.compat.v1.layers.Conv2D(
                activation="relu",
                filters=32,
                kernel_size=[3, 3],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.keras.initializers.GlorotNormal,
                name="conv3",
            )(self.conv2)

            self.flatten = tf.compat.v1.layers.Flatten(
                name="flatten")(self.conv3)

            self.dense = tf.compat.v1.layers.Dense(
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.keras.initializers.GlorotNormal,
                name="fc1",
            )(self.flatten)

            self.output = tf.compat.v1.layers.Dense(
                name="dense",
                units=self.action_size,
                activation=None,
                kernel_initializer=tf.keras.initializers.GlorotNormal,
            )(self.dense)

            self.Q = tf.compat.v1.reduce_sum(
                tf.multiply(self.output, self.action))

            self.loss = tf.compat.v1.reduce_mean(
                tf.square(self.target_q - self.Q))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                self.learning_rate
            ).minimize(self.loss)


""""
    def run(self, inputs_nhwc: np.ndarray):
        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(tf.compat.v1.local_variables_initializer())
            return session.run(self.output, feed_dict={self.inputs: inputs_nhwc})
"""

#net = DQNet((255, 255, 3), 2, 0.008, name="DQNet")
#out = net.run(np.ones((12, 255, 255, 3), dtype=np.float32))
#print(out)
