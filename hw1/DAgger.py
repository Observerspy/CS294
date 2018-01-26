import tensorflow as tf
import os
import numpy as np
import tqdm
import gym
import load_policy

class Config(object):
    n_features = 11
    n_classes = 3
    dropout = 0.5
    hidden_size_1 = 512
    hidden_size_2 = 256
    batch_size = 256
    lr = 0.0001
    lamda = 0.01
    itera = 10
    train_itera = 1000
    envname = 'Hopper-v1'
    max_steps = 1000

class NN(object):
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, Config.n_features), name="input")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, Config.n_classes), name="label")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="drop")
        self.is_training = tf.placeholder(tf.bool)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1, is_training=False):
        if labels_batch is None:
            feed_dict = {self.input_placeholder: inputs_batch,
                         self.dropout_placeholder: dropout, self.is_training: is_training}
        else:
            feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch,
                     self.dropout_placeholder: dropout, self.is_training: is_training}
        return feed_dict

    def add_prediction_op(self):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('w_variable'):
            w1_init = xavier_initializer((Config.n_features, Config.hidden_size_1))
            self.W1 = tf.Variable(w1_init, name="W1")
            tf.summary.histogram("W1", self.W1)
            w2_init = xavier_initializer((Config.hidden_size_1, Config.hidden_size_2))
            self.W2 = tf.Variable(w2_init, name="W2")
            tf.summary.histogram("W2", self.W2)
            w3_init = xavier_initializer((Config.hidden_size_2, Config.n_classes))
            self.W3 = tf.Variable(w3_init, name="W3")
            tf.summary.histogram("W3", self.W3)
        with tf.name_scope('b_variable'):
            b1 = tf.Variable(tf.zeros(Config.hidden_size_1), name="b1")
            tf.summary.histogram("b1", b1)
            b2 = tf.Variable(tf.zeros(Config.hidden_size_2), name="b2")
            tf.summary.histogram("b2", b2)
            b3 = tf.Variable(tf.zeros(Config.n_classes), name="b3")
            tf.summary.histogram("b3", b3)
        self.global_step = tf.Variable(0)

        with tf.name_scope('layer1'):
            layer1 = tf.matmul(self.input_placeholder, self.W1) + b1
            h_batch1 = tf.layers.batch_normalization(layer1, center=True, scale=True, training=self.is_training)
            hidden1 = tf.nn.relu(h_batch1)
            tf.summary.histogram('hidden1_out', hidden1)
            h_drop1 = tf.nn.dropout(hidden1, self.dropout_placeholder)
        with tf.name_scope('layer2'):
            layer2 = tf.matmul(h_drop1, self.W2) + b2
            h_batch2 = tf.layers.batch_normalization(layer2, center=True, scale=True, training=self.is_training)
            hidden2 = tf.nn.relu(h_batch2)
            tf.summary.histogram('hidden2_out', hidden2)
            h_drop2 = tf.nn.dropout(hidden2, self.dropout_placeholder)
        with tf.name_scope('output'):
            pred = tf.matmul(h_drop2, self.W3) + b3
            tf.summary.histogram('out', pred)
        return pred

    def add_loss_op(self, pred):
        loss = tf.losses.mean_squared_error(predictions=pred, labels=self.labels_placeholder)
        loss += Config.lamda * (tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) +
                        tf.nn.l2_loss(self.W3))
        tf.summary.scalar('loss', loss)
        return loss

    def add_training_op(self, loss):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            learning_rate = tf.train.exponential_decay(Config.lr, self.global_step, 1000, 0.8, staircase=True)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, merged, train_writer, i):
        feed = self.create_feed_dict(inputs_batch, labels_batch, self.config.dropout, True)
        rs, _, loss = sess.run([merged, self.train_op, self.loss], feed_dict=feed)
        train_writer.add_summary(rs, i)
        return loss

    def __init__(self, config):
        self.config = config
        self.build()

    def fit(self, sess, train_x, train_y):
        loss = self.train_on_batch(sess, train_x, train_y)

    def build(self):
        with tf.name_scope('inputs'):
            self.add_placeholders()
        with tf.name_scope('predict'):
            self.pred = self.add_prediction_op()
        with tf.name_scope('loss'):
            self.loss = self.add_loss_op(self.pred)
        with tf.name_scope('train'):
            self.train_op = self.add_training_op(self.loss)

    def get_pred(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch, dropout=1, is_training=False)
        p = sess.run(self.pred, feed_dict=feed)
        return p

def load(path):
    all = np.load(path)
    X = all["arr_0"]
    y = all["arr_1"]
    y1 = y.reshape(y.shape[0], y.shape[2])
    return X, y1

def run_env(env, nn,session):
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    observations = []
    while not done:
        action = nn.get_pred(session, obs[None, :])
        observations.append(obs)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        # if args.render:
        #     env.render()
        if steps >= Config.max_steps:
            break
    return totalr, observations

def shuffle(X_train, y_train):
    training_data = np.concatenate((X_train, y_train), axis=1)
    np.random.shuffle(training_data)
    X = training_data[:, :-Config.n_classes]
    y = training_data[:, -Config.n_classes:]
    return X, y


def main():
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(PROJECT_ROOT, "data/"+Config.envname+".train.npz")
    policy_path = os.path.join(PROJECT_ROOT, "experts/"+Config.envname+".pkl")
    train_log_path = os.path.join(PROJECT_ROOT, "log/train/")

    X_train, y_train = load(train_path)#debug

    print("train size :", X_train.shape, y_train.shape)
    print("start training")

    with tf.Graph().as_default():
        config = Config()
        nn = NN(config)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=0.5)

        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(policy_path)
        print('loaded and built')

        with tf.Session() as session:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(train_log_path, session.graph)
            session.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord)

            #iter
            for j in tqdm.tqdm(range(Config.itera)):
                #train
                i = 0
                try:
                    for i in range(Config.train_itera * X_train.shape[0] // Config.batch_size):
                        # something wrong
                        offset = (i * Config.batch_size) % (X_train.shape[0] - Config.batch_size)
                        # shuffle
                        X_train, y_train = shuffle(X_train, y_train)
                        batch_x = X_train[offset:(offset + Config.batch_size), :]
                        batch_y = y_train[offset:(offset + Config.batch_size)]
                        loss = nn.train_on_batch(session, batch_x, batch_y, merged, train_writer, i)
                        i += 1
                    print("step:", i, "loss:", loss)
                    saver.save(session, os.path.join(PROJECT_ROOT, "model/model_ckpt"), global_step=i)
                except tf.errors.OutOfRangeError:
                    print("done")
                finally:
                    coord.request_stop()
                coord.join(threads)

                #get new data and label
                observations = []
                actions = []
                env = gym.make(Config.envname)
                for _ in range(10):
                    _, o = run_env(env, nn, session)
                    observations.extend(o)
                    action = policy_fn(o)
                    actions.extend(action)

                new_x = np.array(observations)
                new_y = np.array(actions)
                X_train = np.concatenate((X_train, new_x))
                y_train = np.concatenate((y_train, new_y))
                print("train size :", X_train.shape, y_train.shape)

                #test
                print("iter:", j, " train finished")
                print(Config.envname + " start")

                rollouts = 10
                returns = []
                for _ in range(rollouts):
                    totalr, _ = run_env(env, nn, session)
                    returns.append(totalr)

                # print('results for ', Config.envname)
                print('returns', returns)
                print('mean return', np.mean(returns), 'std of return', np.std(returns))
                # print('mean return', np.mean(returns))
                print()

if __name__ == '__main__':
    main()