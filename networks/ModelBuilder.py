import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Networks import ModuleFactory


class ModelBuilder:
    def __init__(self, input_window_size: int):
        # model builder context
        self._input_len_meta = input_window_size

        # the input layer of the model, this var is used to generate the tf model obj
        self.input_layer = keras.Input(shape=(input_window_size, ))

        # the latest output of the updated model
        self.model_out = self.input_layer

        # the output_len of the latest updated model
        self.output_shape = (input_window_size, )

        # the blocks counter (will not counting the first input layer)
        self._blocks_counter = 0

    def build(self, network_type: str, layer_size: int, layer_dim_list: list = None, kernel_size_list: list = None,
              filter_list: list = None, stride_list: list = None, activation_list: list = None):
        """
        build one-module of the model, the ModelBuilder will record the building context. While finishing building, call
        to_model() to get the final tf obj.
        :param activation_list: activation function list
        :param kernel_size_list: for conv only, optional. For define the conv kernel size of each conv op.
        :param stride_list: for conv only, optional. For define the stride of each conv op.
        :param filter_list: for conv only, optional. For define the filter size.
        :param network_type: network type, for example: fc, conv, res...
        :param layer_size: layer size for this module to build
        :param layer_dim_list: layer information for this module to build
        :return: this class instance (self ptr)
        """
        if network_type == 'fc':
            module = ModuleFactory.get_fc_blocks(self.output_shape, layer_size, layer_dim_list, activation_list)
        elif network_type == 'conv':
            module = ModuleFactory.get_conv_blocks(self.output_shape, layer_size, kernel_size_list, filter_list,
                                                   stride_list)
        else:
            module = None

        if module is None:
            raise ValueError("no network type [" + network_type + "] is found, please check if the [network_type] " 
                                                                  "parameter is valid")
        else:
            self.model_out = module(self.model_out)
            self.output_shape = self.model_out.shape[1:]
            self._blocks_counter += 1
            return self

    def clear(self):
        """
        clear the building context
        :return: None
        """
        self.input_layer = keras.Input(shape=(self._input_len_meta,))
        self.model_out = self.input_layer
        self.output_shape = (self._input_len_meta, )
        self._blocks_counter = 0

    def to_model(self):
        """
        convert the building context to the tf obj
        :return: tf.keras model obj
        """
        return keras.Model(self.input_layer, self.model_out)

    def print_build_context(self):
        print("===========building context===========")
        print("blocks_counter: " + str(self._blocks_counter))
        print("input_shape: " + str(self.input_layer.shape))
        print("output_shape: " + str(self.model_out.shape))
        print("===========building context===========")


if __name__ == '__main__':
    builder = ModelBuilder(input_window_size=10)
    # model = builder.build('fc', 3, [5, 5, 5]).build('fc', 2, [6, 6]).build('fc', 5, [1, 1, 2, 4, 5]).build('conv', 3).\
    #     build('fc', 2, [10, 1]).to_model()
    # model.summary()
    # builder.print_build_context()
    model = builder.build('fc', 3, [31, 15, 1], activation_list=['relu', 'relu', None])
