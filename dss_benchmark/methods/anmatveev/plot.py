import os

import matplotlib.pyplot as plt

from dss_benchmark.common.preprocess.anmatveev.common import image_path
from dss_benchmark.methods.anmatveev.train import *

__all__ = ["PlotManager"]

fontsize = 14
plt.rcParams.update({"font.size": fontsize})
plt.rcParams["figure.dpi"] = 300
line_thickness = 3


class PlotManager:
    def __init__(self):
        self._ylabel = None
        self._xlabel = None
        self._title = None

    def init_plot(
        self, title, xlabel, ylabel, model, plot_type="F1-score", figsize=(7, 6)
    ):
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._model = model
        self._plot_type = plot_type
        if self._plot_type == "ROC-AUC":
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
        plt.figure(figsize=figsize)
        plt.grid(True)
        plt.title(self._title)
        plt.xlabel(self._xlabel, fontsize=fontsize)
        plt.ylabel(self._ylabel, fontsize=fontsize)

    def add_plot(self, data):
        if self._plot_type == "F1-score":
            if self._model == "word2vec":
                plt.plot(
                    data["steps"],
                    data["thresholds"],
                    label="F1-score(cutoff), sg={}, window={}, epochs={}, min_count={}, vector_size={}".format(
                        data["sg"],
                        data["window"],
                        data["epochs"],
                        data["min_count"],
                        data["vector_size"],
                    ),
                    linewidth=line_thickness,
                )
            elif self._model == "fastText":
                plt.plot(
                    data["steps"],
                    data["thresholds"],
                    label="F1-score(cutoff), sg={}, window={}, epochs={}, min_count={}, vector_size={}, min_n={}, "
                    "max_n={}".format(
                        data["sg"],
                        data["window"],
                        data["epochs"],
                        data["min_count"],
                        data["vector_size"],
                        data["min_n"],
                        data["max_n"],
                    ),
                    linewidth=line_thickness,
                )
            else:
                plt.plot(
                    data["steps"],
                    data["thresholds"],
                    label="F1-score(cutoff)",
                    linewidth=line_thickness,
                )
            plt.plot(
                data["cutoff"],
                data["f1-score"],
                "*",
                label="cutoff={}, max-F1={}".format(data["cutoff"], data["f1-score"]),
            )
        elif self._plot_type == "ROC-AUC":
            if self._model == "word2vec":
                plt.plot(
                    data["fprs"],
                    data["tprs"],
                    linewidth=line_thickness,
                    label="ROC sg={}, window={}, epochs={}, min_count={}, vector_size={}, AUC={}, cutoff={}".format(
                        data["sg"],
                        data["window"],
                        data["epochs"],
                        data["min_count"],
                        data["vector_size"],
                        data["auc"],
                        data["cutoff"],
                    ),
                )
            elif self._model == "fastText":
                plt.plot(
                    data["fprs"],
                    data["tprs"],
                    linewidth=line_thickness,
                    label="ROC sg={},window={},epochs={},min_count={},vector_size={},"
                    "min_n={},"
                    "max_n={},AUC={},cutoff={}".format(
                        data["sg"],
                        data["window"],
                        data["epochs"],
                        data["min_count"],
                        data["vector_size"],
                        data["min_n"],
                        data["max_n"],
                        data["auc"],
                        data["cutoff"],
                    ),
                )
            else:
                plt.plot(
                    data["fprs"],
                    data["tprs"],
                    label="ROC AUC={}, cutoff={}".format(data["auc"], data["cutoff"]),
                    linewidth=line_thickness,
                )
            plt.plot(
                [0, 1], [0, 1], color="navy", linestyle="--", linewidth=line_thickness
            )

    def save(self, imname, legend_loc="lower left", legend_fs=7):
        plt.legend(loc=legend_loc, fontsize=legend_fs)
        dir = os.path.join(image_path, self._model)
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, imname))
