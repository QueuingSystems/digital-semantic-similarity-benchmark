import click
from dss_benchmark.methods.anmatveev.train import *
from dss_benchmark.methods.anmatveev.match import *
from dss_benchmark.methods.anmatveev.plot import *
import pandas as pd


@click.group(
    "research", help="Сопоставление и исследование"
)
def rsch():
    pass


@rsch.command(
    help="Максимизировать F1-score с подбором оптимального cutoff для разных параметров",
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mn", "--model_name", required=True, type=str, help="Путь к модели", prompt=True)
@click.option("-tt", "--train_text", required=True, type=str, help="Путь к обучающему набору", prompt=True)
@click.option("-bt", "--benchmark_text", required=True, type=str, help="Путь к обучающему набору", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def max_f1_ch_best(model_name, train_text, benchmark_text, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    params.texts = train_text
    cases = [
        # windows, epochs, is_sg, min_count, vector_size

        (30, 7, 0, 12, 150),
        # (5, 6, 0, 5, 50),
        # (5, 7, 0, 5, 50),
        # (5, 8, 0, 5, 50),

        # (5, 5, 0, 5, 50),
        # (5, 7, 0, 5, 50),
        # (5, 9, 0, 5, 50),
        # (5, 11, 0, 5, 50),
        #
        # (5, 5, 1, 5, 50),
        # (5, 7, 1, 5, 50),
        # (5, 9, 1, 5, 50),
        # (5, 11, 1, 5, 50),
        #
        # (5, 5, 1, 5, 50),
        # (5, 5, 1, 5, 100),
        # (5, 5, 1, 5, 200),
        # (5, 5, 1, 5, 300)
    ]
    plotManager = PlotManager()
    imname = f"Maximization F1-score-{model_name}"
    plotManager.init_plot(title=imname,
                          xlabel="Cutoff",
                          ylabel="F1-score",
                          figsize=(7, 6))
    benchmark_text = pd.read_json(benchmark_text)
    for case in cases:
        params.window = case[0]
        params.epochs = case[1]
        params.sg = case[2]
        params.min_count = case[3]
        params.vector_size = case[4]

        trainManager = TrainModelManager(params)
        model_path = trainManager.train(model_name)
        matchManager = MatchManager(model_path, params)
        res = matchManager.max_f1(benchmark_text["text_rp"], benchmark_text["text_proj"], benchmark_text, model_path)
        res["window"] = params.window
        res["epochs"] = params.epochs
        res["sg"] = params.sg
        res["min_count"] = params.min_count
        res["vector_size"] = params.vector_size
        plotManager.add_plot(res, plot_type="F1-score")
    imname += f"-best.png"
    plotManager.save(imname)
