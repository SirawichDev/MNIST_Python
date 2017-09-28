import get_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x_point = []
y_point = []
a = 0.22
b = 0.78
for i in range(number_of_points):
    x = np.random.normal(0.0, 0.5)
    y = a * x + b + np.random.normal(0.0, 0.1)

cost_function = tf.reduce_mean(tf.square(y - y_point))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost_function)
mnist_images = get_data.read_data_sets("MNIST_data/", one_hot=False)
train.next_batch(10)
pixels,real_values = mnist_images.train.next_batch(10)
print("list of values loaded ", real_values)
example_to_visualize = 5
print("element N° " + str(example_to_visualize + 1) + " of the list plotted")