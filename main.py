import numpy as np
import tensorflow as tf

# parameters
M = 1000 # max win
batch_size = 128
hidden_size_1 = 128
hidden_size_2 = 128
num_steps = 50000
learning_rate = 1e-3
test_size = 1000

def utility(x):
    return np.log(x+1)/np.log(M+1)

def get_pref_label(c):
    d = c[4]*utility(c[0]) + (1-c[4])*utility(c[1]) - c[5]*utility(c[2]) - (1-c[5])*utility(c[3])
    if (d > 0.05): return [1,0,0]
    elif (d < -0.05): return [0,1,0]
    else: return [0,0,1]

def generate_x(batch_size):
    wins = np.random.rand(batch_size, 4)*M
    probs = np.random.rand(batch_size, 2)
    return np.concatenate((wins, probs), axis=1)

# inputs and targets
X = tf.placeholder(tf.float32, [None, 6])
Y = tf.placeholder(tf.int32, [None, 3])

# net
layer_1 = tf.layers.dense(X, hidden_size_1, activation=tf.nn.relu)
layer_2 = tf.layers.dense(layer_1, hidden_size_2, activation=tf.nn.relu)
out_layer = tf.layers.dense(layer_2, 3)
pred_probs = tf.nn.softmax(out_layer)
pred_class = tf.argmax(out_layer, axis=1)


# loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_layer, labels=tf.argmax(Y, 1)))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(Y, 1), pred_class)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# train
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(num_steps):
    batch_x = generate_x(batch_size)
    batch_y = np.array([get_pref_label(batch_x[i]) for i in range(batch_size)])
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})
    if (i+1) % 1000 == 0:
        loss, acc = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y})
        print('train step: {}, loss: {:.8f}, accuracy: {:.8f}'.format(i+1, loss, acc))

# test
batch_x = generate_x(test_size)
batch_y = np.array([get_pref_label(batch_x[i]) for i in range(test_size)])
loss, acc = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y})
print('test loss: {:.8f}, test accuracy: {:.8f}'.format(loss, acc))