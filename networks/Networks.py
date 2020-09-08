import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
@author chenhao
@desc The basic nn module factory class that uses to implements the nn. We provided the following basic module
that uses to implements the whole nn: 
(1) fc_block: fully-connected block
(2) res_block: basic resnet block
(3) conv_block: one-dim conv block
@date 2020/08/21 
'''


class ModuleFactory:
    """
    factory class used for building basic nn blocks.
    """
    @staticmethod
    def get_fc_blocks(input_shape: tuple, layer_size: int, layer_dim_list: list, activation_list: list = None):
        """
        static factory method to get the fc-block object
        :param activation_list: activation function list
        :param input_shape: the input shape for input the module.
        :param layer_size: the layer size for input the res-nn module.
        :param layer_dim_list: the dimension info for this layer
        :return: Tensorflow-keras nn Object
        """

        if len(layer_dim_list) < layer_size:
            raise ValueError("layer parameter [layer_dim_list] should have the length at least of [layer_size], "
                             "got [layer_dim_list] of length %d, but the [layer_size] is %d", len(layer_dim_list),
                             layer_size)
        if activation_list is None:
            activation_list = ['relu'] * layer_size

        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))

        # for non-vector type tensor, we need to flatten the tensor to one-dim vector
        if len(input_shape) >= 2:
            model.add(layers.Flatten())

        for i in range(layer_size):
            model.add(layers.BatchNormalization())
            if activation_list[i] == 'none':
                model.add(layers.Dense(units=layer_dim_list[i]))
            else:
                model.add(layers.Dense(units=layer_dim_list[i], activation=activation_list[i]))

        return model

    @staticmethod
    def get_res_blocks(input_shape: tuple, layer_size: int, layer_dim_list: list):
        """
        static factory method to get the res-block object
        :param layer_dim_list: the dimension info for this layer
        :param input_shape: the input shape for inputting the res-nn module.
        :param layer_size: the layer size for input the res-nn module.
        :return: Tensorflow-keras nn Object
        """
        inp = layers.Input(shape=input_shape)
        x = inp
        for i in range(layer_size):
            a = layers.Dense(units=layer_dim_list[i], activation='relu')(x)
            y = layers.Add()([x, a])
            x = y

        model = keras.Model(inp, x)
        return model

    @staticmethod
    def get_conv_blocks(input_shape: tuple, layer_size: int, kernel_size_list: list = None, filter_list: list = None,
                        stride_list: list = None):
        """
         static factory method to get the 1-dim-conv-block object
        :param input_shape: input-dim for inputting this module
        :param layer_size: layer size of this module
        :param stride_list: conv stride list
        :param kernel_size_list: conv kernel size list
        :param filter_list: filter num list
        :return: Tensorflow nn object
        """
        if stride_list is None:
            stride_list = [1] * layer_size

        if filter_list is None:
            filter_list = [16] * layer_size

        if kernel_size_list is None:
            kernel_size_list = [1] * layer_size

        inp = layers.Input(shape=input_shape)

        if len(input_shape) == 1:
            x = tf.expand_dims(inp, axis=-1)
        else:
            x = inp

        for i in range(layer_size):
            x = layers.Conv1D(filter_list[i], kernel_size_list[i], stride_list[i], padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        return keras.Model(inp, x)


class CriticFactory:
    """
    factory class for building critics
    """
    @staticmethod
    def get_fc_critic(input_len: int, layer_size: int, layer_dim_list: list):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_len, )))
        for i in range(layer_size):
            model.add(layers.Dense(units=layer_dim_list[i]))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())

        model.add(layers.Activation('sigmoid'))

    @staticmethod
    def get_conv_critic(input_len: int, layer_size: int, kernel_size_list: list = None, filter_list: list = None,
                        stride_list: list = None):
        if stride_list is None:
            stride_list = [1] * layer_size

        if filter_list is None:
            filter_list = [16] * layer_size

        if kernel_size_list is None:
            kernel_size_list = [3] * layer_size

        inp = layers.Input(shape=(input_len, ))
        x = tf.expand_dims(inp, axis=-1)

        for i in range(layer_size):
            x = layers.Conv1D(filter_list[i], kernel_size_list[i], stride_list[i], padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        x = layers.Conv1D(1, 3, 1, padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=1, activation='sigmoid')(x)

        return keras.Model(inp, x)


if __name__ == '__main__':
    model = CriticFactory.get_conv_critic(3200, 5)
    model.summary()