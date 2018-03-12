import tensorflow as tf
import os
import numpy as np


class Estimator:
    """ Class for both Q-Network and Target Network"""

    def __init__(self, param, scope="estimator"):
        self.d_in = param['d_in']
        self.d_out = param['d_out']
        self.h = param['h']
        self.sess = param['sess']
        self.global_step_tensor = param['global_step']
        self.tf_summary_dir = param['tf_summary_dir']
        self.scope = scope

        # Placeholders
        self.X = None
        self.Y = None
        self.A = None
        self.w = None
        self.predictions = None
        self.td_error = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.summaries = None
        self.loss_sl = None
        self.loss_rl = None

        self.margin_loss = None
        self.sl_actions = None
        self.sl_state = None

        self.summary_writer = None
        self.lr = None

        with tf.variable_scope(scope):
            self.build_model()
            if self.tf_summary_dir:
                summary_dir = os.path.join(self.tf_summary_dir, "summaries_{}".
                                           format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def build_model(self):
        """Builds the TensorFlow Computational Graph"""
        lr = 1e-3  # Learning rate
        lambda_reg = 1e-1
        lambda_sl = 1
        num_layers = 1

        regularizer = tf.contrib.layers.l2_regularizer(scale=lambda_reg)

        self.X = tf.placeholder(name='X', dtype=tf.float32, shape=[None,
                                                                   self.d_in])
        self.Y = tf.placeholder(name='Y', dtype=tf.float32, shape=[None])  #
        self.A = tf.placeholder(name='A', dtype=tf.int32, shape=[None])
        self.sl_actions = tf.placeholder(name='SL_A', dtype=tf.int32, shape=[
            None])
        self.margin_loss = tf.placeholder(name='L', dtype=tf.float32, shape=[
            None, self.d_out])
        self.w = tf.placeholder(name='W', dtype=tf.float32, shape=[None])
        self.lr = tf.placeholder(name='lr', dtype=tf.float32, shape=[])

        batch_size = tf.shape(self.X)[0]
        demo_size = tf.shape(self.sl_actions)[0]

        hidden = self.X
        for _ in range(num_layers):
            hidden = tf.layers.dense(hidden, self.h, activation=tf.nn.relu,
                                     kernel_regularizer=regularizer)

        self.predictions = tf.layers.dense(hidden, self.d_out, name='output',
                                           kernel_regularizer=regularizer)

        q_rl = self.predictions[:batch_size, :]
        q_sl = self.predictions[batch_size:, :]

        gather_indices = tf.range(0, batch_size) * self.d_out + self.A
        Y_pred = tf.gather(tf.reshape(q_rl, [-1]), gather_indices)
        q_pred_sl = tf.reduce_max(q_sl + self.margin_loss, axis=1)
        gather_indices = tf.range(0, demo_size) * self.d_out + self.sl_actions
        q_opt_sl = tf.gather(tf.reshape(q_sl, [-1]), gather_indices)

        self.td_error = tf.abs(self.Y - Y_pred)
        self.loss_rl = tf.reduce_mean(tf.multiply(self.td_error ** 2, self.w))
        self.loss_sl = tf.reduce_mean(q_pred_sl-q_opt_sl)
        self.loss = self.loss_rl + lambda_sl * self.loss_sl

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=self.global_step_tensor)

        self.summaries = tf.summary.merge(
            [tf.summary.scalar("loss", self.loss),
             tf.summary.histogram("q_values_hist", self.predictions),
             tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))]
        )

    def predict(self, state):
        """Perform a forward pass to predict Q-values for the state"""
        return sess.run(self.predictions, {self.X: state})

    def get_loss(self, state, action, y_target, weights_is,
                 margin_loss, sl_actions, batch_size, demo_size):
        feed_dict = {self.X: state, self.A: action, self.Y: y_target,
                     self.w: weights_is, self.margin_loss: margin_loss,
                     self.sl_actions: sl_actions, self.batch_size:
                         batch_size, self.demo_size: demo_size}
        loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss

    def update(self, inputs, save_graph=False):
        """Update the estimator"""
        global_step = tf.train.global_step(self.sess, self.global_step_tensor)

        feed_dict = {}
        pass
        feed_dict[self.X] = inputs['state']
        feed_dict[self.A] = inputs['action']
        feed_dict[self.Y] = inputs['y_target']
        feed_dict[self.w] = inputs['weights_is']
        feed_dict[self.margin_loss] = inputs['margin_loss']
        feed_dict[self.sl_actions] = inputs['sl_actions']


        feed_dict = {self.X: state, self.A: action, self.Y: y_target,
                     self.w: weights_is, self.margin_loss: margin_loss,
                     self.sl_actions: sl_actions, self.batch_size:
                         batch_size, self.demo_size: demo_size, self.lr: lr}
        summaries, _, loss, td_error, loss_sl = self.sess.run(
            [self.summaries, self.train_op, self.loss, self.td_error,
             self.loss_sl],
            feed_dict=feed_dict)

        print("Supervised Loss: ", loss_sl)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        if save_graph:
            self.summary_writer.add_graph(self.sess.graph)

        return loss, td_error
