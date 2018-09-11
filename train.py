import pandas as pd
import random as rd
import tensorflow as tf
import model as mlr

NUM_FEATURES = 2
NUM_SEPARATIONS = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.001
STEPS =100000
DECAY_STEP = 1000
DECAY_RATE = 0.98

data = pd.read_csv('data.csv', sep=' ', header=None)

def get_random_batch(batch_size=10):
    max_len = data.shape[0]
    index = rd.randint(0, max_len-batch_size+1)
    batch_x = data.values[index:index+batch_size,0:-1]
    batch_y = data.values[index:index+batch_size,-1:]
    #print(batch_x)
    #print(batch_y)
    return batch_x, batch_y

input_tensor = tf.placeholder(tf.float32, shape=(None, NUM_FEATURES)) 
gt_tensor = tf.placeholder(tf.float32, shape=(None, 1))

output_tensor = mlr.model(input_tensor, NUM_FEATURES, NUM_SEPARATIONS)
loss = tf.reduce_mean(tf.square(tf.subtract(gt_tensor,output_tensor)))
global_step = tf.Variable(0, trainable=False)
learning_rate   = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEP, DECAY_RATE, staircase=False)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt = optimizer.minimize(loss, global_step=global_step)

saver = tf.train.Saver()


with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    try:
        saver.restore(sess, sys.argv[1])
    except:
        pass

    for step in range(STEPS):
        batch_x, batch_y = get_random_batch(batch_size=BATCH_SIZE)
        ot,lr,l,_ = sess.run([output_tensor, learning_rate, loss, opt], feed_dict={input_tensor: batch_x, gt_tensor: batch_y})
        if step%2000==0:
            print("loss: "+str(l)+" Learning rate: "+str(lr))
        if step%10000==0:
            saver.save(sess, "./saved/model.ckpt")
