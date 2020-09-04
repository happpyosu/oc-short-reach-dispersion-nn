import matplotlib.pyplot as plt


class PlotUtils:
    """
    plotUtils offer wrapped plot methods for plotting tf tensors or numpy ndarray
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

    @staticmethod
    def plot_eye_diagram(wave: list, sample_per_sym: int, offset: int, sym_len: int, is_save_file: bool = False,
                         file_name: str = 'eye_diagram.jpg'):
        """
        plot the eye diagram of a 1D wave signal
        :param file_name:
        :param is_save_file: save the file or not
        :param sample_per_sym: sample per symbols
        :param wave: wave sample point list
        :param offset: offset of the start point
        :param sym_len: symbol length of the eye diagram window
        :return:
        """
        eye_win = sample_per_sym * sym_len
        if len(wave) < eye_win:
            raise ValueError("[plot_eye_diagram]: the length of [wave] list " + str(len(wave)) +
                             "is shorter that the eye windows")
        plt.figure()
        cur = 0

        while cur + eye_win <= len(wave):
            plt.plot(wave[cur:cur + eye_win])
            cur = cur + eye_win

        if is_save_file:
            plt.savefig('../output/' + file_name)

        plt.show()
