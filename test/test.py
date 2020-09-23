import sys
import tensorflow as tf

a = [1, 2, 3]

hot = tf.one_hot(a, 4)
print(hot)
