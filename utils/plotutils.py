import matplotlib.pyplot as plt


class PlotUtils:
    """
    plotUtils offer wrapped plot methods for plotting tf tensors
    """
    @staticmethod
    def plot_wave_tensors(*wave_tensors, legend_list: list = None, is_save_file=False, file_name='wave.jpg'):
        """
        static method used to plot the 1D signals, the arg should be a 1D-signal (batch-size exclusive)
        :param file_name: saving file name
        :param is_save_file: save the file or not
        :param legend_list: legend list for marking in the plot, if is None, auto-generate increasing legends
        :param wave_tensors: tf tensors (batch-size exclusive)
        :return: None
        """
        if len(wave_tensors) != len(legend_list):
            legend_list = [str(i) for i in range(len(wave_tensors))]
        index = 0
        plt.figure()

        for tensor in wave_tensors:
            tensor_numpy = tensor.numpy()
            plt.plot(tensor_numpy)
            index += 1

        plt.legend(legend_list)
        if is_save_file:
            plt.savefig('../output/' + file_name)
        plt.show()


