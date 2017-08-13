import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.python.ops import rnn

import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        self.args.data_dim = 1
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        one_cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([one_cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.float32, \
                [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.float32, \
                [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        inputs = tf.split(1, args.seq_length, self.input_data)

        outputs, last_state = \
                rnn.rnn(cell, inputs, self.initial_state,
                        dtype=tf.float32)
        self.final_state = last_state

        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])

        softmax_w = tf.get_variable("softmax_w", \
                [args.rnn_size, self.args.data_dim])
        softmax_b = tf.get_variable("softmax_b", \
                [self.args.data_dim])

        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        flat_targets = tf.reshape(self.targets, [-1])
        print(self.logits)
        print(self.targets)
        self.cost = tf.reduce_sum(tf.pow(self.logits-flat_targets, 2))/ \
                (2*(args.batch_size*args.seq_length)) #L2 loss
        print(self.cost)
        self.lr = tf.Variable(args.learning_rate, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        #optimizer = tf.train.AdamOptimizer(self.lr)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdagradOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, num=200, prime=[0.0]):
        state = self.cell.zero_state(1, tf.float32).eval()
        for price in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = price
            feed = {self.input_data: x, self.initial_state:state}
            [state, logits] = \
                    sess.run([self.final_state, self.logits], feed)
            #print price, logits

        ret = prime
        price = prime[-1]
        for n in xrange(num):
            x = np.zeros((1, 1))
            x[0, 0] = price
            feed = {self.input_data: x, self.initial_state:state}
            [logits, state] = \
                    sess.run([self.logits, self.final_state], feed)
            print logits
            import sys; sys.exit();
            pred = logits
            ret += [pred]
            price = pred
return ret
