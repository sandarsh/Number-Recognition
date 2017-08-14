'''
Input data > weights > hiddenlayer1 (activation functions) > weights > hiddenlayer2 (activation function) > weights > output layer
Compare out put to intended output > cost functions (cross entropy)
optimization function (optimizer) > minimise cost (Adam Optimizer, SGD, AdaGrad)
backpropagation
feed forward + backpropagation = epoch

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/temp/data/", one_hot=True)

n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500
n_classes = 10
batch_size = 100

#height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([784, n_nodes_h1])) ,
                      'biases' : tf.Variable(tf.random_normal([n_nodes_h1]))}
    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_h2]))}
    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])) , 'biases' : tf.Variable(tf.random_normal([n_nodes_h3]))}
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_h3, n_classes])) , 'biases' : tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1);

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2);

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3);

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


def train_neural_network(x,y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in xrange(epochs):
            epoch_loss = 0
            for _ in xrange(int(mnist.train.num_examples/batch_size)):
                ep_x,ep_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x:ep_x, y:ep_y})
                epoch_loss += c
            print("Epoch", epoch , "of", epochs, "Loss", epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print ("Accuracy:", accuracy.eval({x : mnist.test.images, y: mnist.test.labels}))


train_neural_network(x,y)
