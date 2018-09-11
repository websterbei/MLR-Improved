import pandas as pd
import random as rd
import tensorflow as tf
import model as mlr
import sys

NUM_FEATURES = 2
NUM_SEPARATIONS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.001
STEPS =100000

data = pd.read_csv('data_test.csv', sep=' ', header=None)

input_tensor = tf.placeholder(tf.float32, shape=(1, NUM_FEATURES)) 

output_tensor = mlr.model(input_tensor, NUM_FEATURES, NUM_SEPARATIONS)

saver = tf.train.Saver()

with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    saver.restore(sess, sys.argv[1])
    for i in range(data.shape[0]):
        ot = sess.run([output_tensor], feed_dict={input_tensor: data.values[i:i+1,0:NUM_FEATURES]})
        print ot[0][0][0]
    saver.save(sess, "./saved/model.ckpt")
