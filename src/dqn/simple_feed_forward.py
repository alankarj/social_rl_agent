import tensorflow as tf
import numpy as np
import math

class FeedForward:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.session = None
        self.X = None
        self.Y = None
        self.loss = None

        self.create_model()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        d_in = self.input_dim
        h = self.hidden_dim
        d_out = self.output_dim

        W1 = tf.get_variable("W1", shape=[d_in, h],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[h])

        W2 = tf.get_variable("W2", shape=[h, d_out],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", shape=[d_out])

        out1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)
        self.Y = tf.matmul(out1, W2) + b2

        # self.loss = tf.reduce_mean(tf.losses.mean_squared_error(Y_pred, Y_true))

    # def prepare_data(self, data, clone_dqn):
    #     S = np.array([dp[0] for dp in data])
    #     S_tplus1 = np.array([dp[3] for dp in data])
    #
    #     Y_full_train = self.get_Y(S)
    #     Y_full_target = clone_dqn.get_Y(S_tplus1)
    #
    #     Y_final_train = []
    #     Y_final_target = []
    #
    #     for i, d in enumerate(data):
    #         action = d[1]
    #         over = d[4]
    #         reward = d[2]
    #         Y_final_train.append(Y_full_train[i][action])
    #
    #         y_target = reward
    #         max_action = np.nanargmax(Y_full_target[i])
    #         if not over:
    #             y_target += Y_full_target[i][max_action]
    #
    #         Y_final_target.append(y_target)
    #
    #     Y_pred = tf.convert_to_tensor(Y_final_train)
    #     Y_true = tf.convert_to_tensor(Y_final_target)
    #
    #     return Y_pred, Y_true
    #
    # def run_model(self, session, loss, X, Y, batch_size, num_iter):
    #     data_size = len(X)
    #
    #     num_batches = int(math.ceil(data_size/batch_size))
    #     train_indicies = np.arange(data_size)
    #     np.random.shuffle(train_indicies)
    #
    #     for it in range(num_iter):
    #         for i in range(num_batches):
    #             start_idx = (i*batch_size) % data_size
    #             ids = train_indicies[start_idx:start_idx + batch_size]
    #
    #             feed_dict = {self.X: X[ids, :]}
    #             loss = self.session.run(self.loss, feed_dict)
    #             print("Loss: ", loss)
    #
    # def train(self, data, batch_size, clone_dqn):
    #     learning_rate = 1e-3
    #
    #     Y_pred, Y_true = self.prepare_data(data, clone_dqn)
    #
    #
    #     optimizer = tf.train.AdamOptimizer(learning_rate)
    #     train_step = optimizer.minimize(self.loss)
    #
    # def get_Y(self, Xd):
    #     feed_dict = {self.X: Xd}
    #     return self.session.run(self.Y, feed_dict)
    #
    # def predict(self, state_representation):
    #     feed_dict = {self.X: state_representation}
    #     y = self.session.run(self.Y, feed_dict)
    #     return np.argmax(y)


