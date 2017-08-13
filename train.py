import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from utils import DataLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/maraoz/shovestore',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=100,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = DataLoader(args.data_dir, args.batch_size, args.seq_length)

    with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in xrange(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                #print(x, '->', y)
                #import sys; sys.exit();
                feed = {
                    model.input_data: x, 
                    model.targets: y,
                    model.initial_state: state
                }
                train_loss, state, _ = sess.run(\
                        [model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start)
                if (e * data_loader.num_batches + b) % args.save_every == 0:
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print "model saved to {}".format(checkpoint_path)

if __name__ == '__main__':
main()
