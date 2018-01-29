import tensorflow as tf
import numpy as np
import math
# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.mean_s, self.std_s, self.mean_deltas, self.std_deltas, self.mean_a, self.std_a = normalization
        self.sess = sess
        self.batch_size = batch_size
        self.iter = iterations
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.s_a = tf.placeholder(shape=[None, self.s_dim + self.a_dim], name="s_a", dtype=tf.float32)
        self.deltas = tf.placeholder(shape=[None, self.s_dim], name="deltas", dtype=tf.float32)
        self.deltas_predict = build_mlp(self.s_a, self.s_dim, "NND", n_layers=n_layers, size=size,
                                 activation=activation, output_activation=output_activation)

        self.loss = tf.reduce_mean(tf.square(self.deltas_predict - self.deltas))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    #train deltas(s-sp) perdict deltas
    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states,
        (unnormalized)actions, (unnormalized)next_states and fit the dynamics model
        going from normalized states, normalized actions to normalized state differences
        (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        s = np.concatenate([d["state"] for d in data])
        sp = np.concatenate([d["next_state"] for d in data])
        a = np.concatenate([d["action"] for d in data])
        N = s.shape[0]
        train_indicies = np.arange(N)

        #normalize
        s_norm = (s - self.mean_s) / (self.std_s + 1e-7)
        a_norm = (a - self.mean_a) / (self.std_a + 1e-7)
        s_a = np.concatenate((s_norm, a_norm), axis=1)
        deltas_norm = ((sp - s) - self.mean_deltas) / (self.std_deltas + 1e-7)

        #train
        for j in range(self.iter):
            np.random.shuffle(train_indicies)
            for i in range(int(math.ceil(N / self.batch_size))):
                start_idx = i * self.batch_size % N
                idx = train_indicies[start_idx:start_idx + self.batch_size]
                batch_x = s_a[idx, :]
                batch_y = deltas_norm[idx, :]
                self.sess.run([self.train_op], feed_dict={self.s_a: batch_x, self.deltas: batch_y})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states
        and (unnormalized) actions and return the (unnormalized) next states
        as predicted by using the model """
        """ YOUR CODE HERE """
        #normalize
        s_norm = (states - self.mean_s) / (self.std_s + 1e-7)
        a_norm = (actions - self.mean_a) / (self.std_a + 1e-7)
        s_a = np.concatenate((s_norm, a_norm), axis=1)

        delta = self.sess.run(self.deltas_predict, feed_dict={self.s_a: s_a})

        #denormalize
        return delta * self.std_deltas + self.mean_deltas + states



