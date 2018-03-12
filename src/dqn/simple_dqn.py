from dqn.estimator import Estimator
import os
import tensorflow as tf
import numpy as np


class SimpleDQN:

    def __init__(self, d_in, d_out, h, params):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.tf_summary_dir = os.path.abspath('./tf-summaries/')
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.d_in = d_in
        self.d_out = d_out
        self.h = h

        param_est = self.get_param_estimator()
        self.q_estimator = Estimator(param_est, scope="q-network")
        self.target_estimator = Estimator(param_est, scope="target-q-network")

        self.model_dir = os.path.abspath('./models/')
        self.gamma = params['gamma']
        self.L = params['L']

        # Variables for supervised learning (training) data
        self.sl_states = None
        self.sl_actions = None
        self.all_possible_actions = None  # All actions off just by the CS
        self.num_erp_sl = None  # Size of the ERP in SL

        # Variables for supervised learning (test) data
        self.test_sl_states = None
        self.test_sl_actions = None
        self.test_all_possible_actions = None
        self.test_num_erp_sl = None

        self.sess.run(tf.global_variables_initializer())

    def get_param_estimator(self):
        param_est = {}
        pass
        param_est['d_in'] = self.d_in
        param_est['d_out'] = self.d_out
        param_est['h'] = self.h
        param_est['sess'] = self.sess
        param_est['global_step'] = self.global_step
        param_est['tf_summary_dir'] = self.tf_summary_dir
        return param_est

    def train(self, sample, weights_is, lr):
        state, action, reward, next_state, over = zip(*sample)
        batch_size = len(state)
        demo_size = self.num_erp_sl
        margin_loss = self.get_margin_loss()
        state = np.vstack((state, self.sl_states))

        q_values_next = self.target_estimator.predict(next_state)
        y_target = reward + np.invert(over).astype(np.float32) * self.gamma *\
                        np.amax(q_values_next, axis=1)

        loss, td_error = self.q_estimator.update(state, action,
                                                 y_target, weights_is,
                                                 margin_loss, self.sl_actions,
                                                 batch_size, demo_size, lr)
        return loss, td_error

    def test(self, sample, lr):
        state, action, reward, next_state, over = zip(*sample)
        batch_size = len(state)
        demo_size = self.test_num_erp_sl
        weights_is = [1]*batch_size
        margin_loss = self.get_margin_loss(test=True)
        state = np.vstack((state, self.test_sl_states))

        q_values_next = self.target_estimator.predict(next_state)
        y_target = reward + np.invert(over).astype(np.float32) * self.gamma *\
                            np.amax(q_values_next, axis=1)

        loss = self.q_estimator.get_loss(state, action, y_target,
                                         weights_is, margin_loss,
                                         self.test_sl_actions, batch_size,
                                         demo_size)
        return loss

    def update_target_q_network(self):
        estimator1 = self.q_estimator
        estimator2 = self.target_estimator

        e1_params = [t for t in tf.trainable_variables() if
                     t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if
                     t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)
        update_ops = []

        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)
        self.sess.run(update_ops)

    def get_best_action(self, state):
        state = np.reshape(np.array(state), (1, len(state)))
        y = self.q_estimator.predict(state)
        return np.argmax(y)

    def set_sl_loss(self, sl_states, sl_actions, all_possible_actions,
                    test=False):
        if test:
            self.test_sl_states = sl_states
            self.test_sl_actions = sl_actions
            self.test_all_possible_actions = all_possible_actions
            self.test_num_erp_sl = len(self.test_sl_states)
        else:
            self.sl_states = sl_states
            self.sl_actions = sl_actions
            self.all_possible_actions = all_possible_actions
            self.num_erp_sl = len(self.sl_states)

    def get_margin_loss(self, test=False):
        if test:
            L_mat = np.ones((self.test_num_erp_sl, self.d_out)) * self.L
            L_mat[self.test_all_possible_actions] = 0
        else:
            L_mat = np.ones((self.num_erp_sl, self.d_out))*self.L
            L_mat[self.all_possible_actions] = 0
        return L_mat

    def save_model(self, gs, file_name, sl=True):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, file_name)
        saver = tf.train.Saver(max_to_keep=100)
        if not sl:
            saver.save(self.sess, model_path, write_meta_graph=True)
        else:
            saver.save(self.sess, model_path, global_step=self.global_step,
                       write_meta_graph=True)

    def close_session(self):
        self.sess.close()

    def restore_session(self, mfile):
        sess = tf.Session()
        saver = tf.train.import_meta_graph(self.model_dir + mfile + '.meta')
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.model_dir + mfile)
        self.sess = sess
