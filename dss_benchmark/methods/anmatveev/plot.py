import os

import matplotlib.pyplot as plt
from dss_benchmark.methods.anmatveev.train import *

fontsize = 14
plt.rcParams.update({'font.size': fontsize})
plt.rcParams['figure.dpi'] = 300
line_thickness = 3


class PlotManager:
    def __init__(self):
        self._ylabel = None
        self._xlabel = None
        self._title = None

    def init_plot(self,
                  title,
                  xlabel,
                  ylabel,
                  model,
                  figsize=(7, 6)):
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._model = model
        plt.figure(figsize=figsize)
        plt.grid(True)
        plt.title(self._title)
        plt.xlabel(self._xlabel, fontsize=fontsize)
        plt.ylabel(self._ylabel, fontsize=fontsize)

    def add_plot(self, data, plot_type="F1-score"):
        if plot_type == "F1-score":
            if self._model == "word2vec":
                plt.plot(data["steps"], data["thresholds"],
                         label="F1-score(cutoff), window={}, epochs={}, sg={}, min_count={}, vector_size={}".format(
                             data["window"], data["epochs"], data["sg"], data["min_count"], data["vector_size"]
                         ),
                         linewidth=line_thickness)
            elif self._model == "fastText":
                plt.plot(data["steps"], data["thresholds"],
                         label="F1-score(cutoff), window={}, epochs={}, sg={}, min_count={}, vector_size={}, min_n={}, "
                               "max_n={}".format(
                             data["window"], data["epochs"], data["sg"], data["min_count"], data["vector_size"],
                             data["min_n-max_n"][0], data["min_n-max_n"][1]
                         ),
                         linewidth=line_thickness)
            plt.plot(data["cutoff"], data["f1-score"], "*", label="cutoff={}, max-F1={}".format(data["cutoff"], data["f1-score"]))
        elif plot_type == "ROC-AUC":
            pass

    def save(self, imname=None, image_path=None, legend_loc="lower left", legend_fs=7):
        plt.legend(loc=legend_loc, fontsize=legend_fs)
        if not image_path:
            dir = os.path.join(image_path, self._model)
            os.makedirs(dir, exist_ok=True)
            plt.savefig(os.path.join(dir, imname))
        else:
            plt.savefig(image_path)