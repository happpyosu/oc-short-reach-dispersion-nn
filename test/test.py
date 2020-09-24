import sys
import tensorflow as tf

a = tf.constant([[1]], dtype=tf.float32)
a = tf.squeeze(a, axis=0).numpy()[0]
print(a == 1)