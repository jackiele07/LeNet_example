from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))


index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 10
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Store layers weight & bias
    weights = {
        'w1': tf.Variable(tf.random_normal([5, 5, 3, 6])),
        'w2': tf.Variable(tf.random_normal([5, 5, 6, 16])),
        'w_fc1': tf.Variable(tf.random_normal([5*5*16, 120])),
        'w_fc2': tf.Variable(tf.random_normal([120, 84])),
        'out': tf.Variable(tf.random_normal([84, 10]))}

    biases = {
        'b1': tf.Variable(tf.random_normal([6])),
        'b2': tf.Variable(tf.random_normal([16])),
        'b_fc1': tf.Variable(tf.random_normal([120])),
        'b_fc2': tf.Variable(tf.random_normal([84])),
        'b_out': tf.Variable(tf.random_normal([10]))}

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer1 = tf.nn.conv2d(x,weights['w1'],strides=[1,1,1,1],padding='VALID')
    layer1 = tf.nn.bias_add(layer1,biases['b1'])

    # TODO: Activation.
    layer1 = tf.nn.relu(layer1)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    layer1 = tf.nn.max_pool(layer1,ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    layer2 = tf.nn.conv2d(layer1,weights['w2'], strides=[1,1,1,1],padding='VALID')
    layer2 = tf.nn.bias_add(layer2,biases['b2'])
    # TODO: Activation.
    layer2 = tf.nn.relu(layer2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2 = tf.nn.max_pool(layer2,ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer_fc1 = tf.reshape(layer2, [-1, weights['w_fc1'].get_shape().as_list()[0]])
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    layer_fc1 = tf.add(tf.matmul(layer_fc1,weights['w_fc1']), biases['b_fc1'])
    # TODO: Activation.
    layer_fc1 = tf.nn.relu(layer_fc1)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    layer_fc2 = tf.add(tf.matmul(layer_fc1, weights['w_fc2']), biases['b_fc2'])
    # TODO: Activation.
    layer_fc2 = tf.nn.relu(layer_fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(layer_fc2, weights['out']), biases['b_out'])
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
