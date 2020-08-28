import tensorflow as tf

if __name__ == '__main__':
    a = tf.constant([[[1., 2., 3.]]], dtype=tf.float32)
    b = tf.constant([[[1., 1., 1.]]], dtype=tf.float32)

    mse = tf.keras.losses.MSE(a, b)
    print(a.shape)
    print(mse.shape)