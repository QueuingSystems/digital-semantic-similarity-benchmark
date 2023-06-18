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
                  figsize=(7, 6)):
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        plt.figure(figsize=figsize)
        plt.grid(True)
        plt.title(self._title)
        plt.xlabel(self._xlabel, fontsize=fontsize)
        plt.ylabel(self._ylabel, fontsize=fontsize)

    def add_plot(self, data, plot_type="F1-score"):
        if plot_type == "F1-score":
            plt.plot(data["steps"], data["thresholds"],
                     label="F1-score(cutoff), window={}, epochs={}, sg={}, min_count={}, vector_size={}".format(
                         data["window"], data["epochs"], data["sg"], data["min_count"], data["vector_size"]
                     ),
                     linewidth=line_thickness)
            plt.plot(data["cutoff"], data["f1-score"], "*", label="(cutoff, max-F1)({}, {})".format(data["cutoff"], data["f1-score"]))
        elif plot_type == "ROC-AUC":
            pass

    def save(self, imname, legend_loc="lower left", legend_fs=8):
        plt.legend(loc=legend_loc, fontsize=legend_fs)
        plt.savefig(image_path + imname)
