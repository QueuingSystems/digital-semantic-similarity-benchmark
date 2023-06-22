import os

import click
from dss_benchmark.methods.anmatveev.train import *
from dss_benchmark.methods.anmatveev.match import *
from dss_benchmark.methods.anmatveev.plot import *
from dss_benchmark.methods.anmatveev.params_parser import *
import pandas as pd

PARAMS_SEQUENCE_WORD2VEC = ["window", "epochs",  "sg", "min_count", "vector_size"]
PARAMS_SEQUENCE_FASTTEXT = PARAMS_SEQUENCE_WORD2VEC[:] + ["min_n-max_n"]
@click.group(
    "research", help="Сопоставление и исследование"
)
def rsch():
    pass


@rsch.command(
    help="Обучить все возможные комбинации моделей",
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("-fp", "--file_path", required=True, type=str, help="Путь к файлу с параметрами", prompt=True)
@click.option("-tr", "--train_text", required=True, type=str, help="Обучающий набор", prompt=True)
@click.option("-mp", "--models_path", required=True, type=str, help="Путь к моделям", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
# pretrain models before its comparing with splitting into groups
def train_cascade(file_path, train_text, models_path, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    params.texts = train_text
    os.makedirs(models_path, exist_ok=True)
    parser = ParamsParser(file_path)
    groups = parser.read(split_into_groups=True)
    for group in groups:
        subdir = os.path.join(models_path, group)
        os.makedirs(subdir, exist_ok=True)
        i = 1
        for case in groups[group]:
            params.window = case[1][1]
            params.epochs = case[1][2]
            params.sg = case[1][3]
            params.min_count = case[1][4]
            params.vector_size = case[1][5]
            if "fastText" in case[0]:
                params.min_n = case[1][6]
                params.max_n = case[1][7]
            trainManager = TrainModelManager(params)
            trainManager.train(case[1][0], model_path=os.path.join(subdir, str(i) + '-' + case[0]))
            i += 1
            print('case: ', case[0])


@rsch.command(
    help="Подбор оптимальных параметров для максимизации F1-score",
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mp", "--models_path", required=True, type=str, help="Путь к папке с моделями", prompt=True)
@click.option("-bt", "--benchmark_text", required=True, type=str, help="Путь к бенчмарку", prompt=True)
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.option("-bpp", "--best_params_path", required=True, type=str, help="Путь к файлу с оптимальными параметрами", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
# cases in a file will be splitted into groups with a space
def get_best_params_f1(models_path, benchmark_text, text1, text2, best_params_path, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    benchmark_text = pd.read_json(benchmark_text)
    best_params_list = []
    group_number = 1
    model_name = models_path.split('/')[-1]

    for group, dirs, files in sorted(os.walk(models_path))[1:]:
        plotManager = PlotManager()
        imname = f"Maximization-F1-score-{model_name}"
        plotManager.init_plot(title=imname,
                              xlabel="Cutoff",
                              ylabel="F1-score",
                              model=model_name,
                              figsize=(7, 6))
        max_f1 = 0
        best_param = 0
        for file in sorted(filter(lambda f: '.wv.vectors_ngrams.npy' not in f, files)):
            print(file)
            model_path = os.path.join(group, file)
            case = ParamsParser().read_one(file)
            params.window = case[1]
            params.epochs = case[2]
            params.sg = case[3]
            params.min_count = case[4]
            params.vector_size = case[5]
            if "fastText".lower() in models_path.lower():
                params.min_n = case[6]
                params.max_n = case[7]
            
            matchManager = MatchManager(model_path, params)
            res = matchManager.max_f1(benchmark_text[text1], benchmark_text[text2], benchmark_text,
                                      model_path)

            res["window"] = params.window
            res["epochs"] = params.epochs
            res["sg"] = params.sg
            res["min_count"] = params.min_count
            res["vector_size"] = params.vector_size
            res["min_n-max_n"] = (params.min_n, params.max_n)
            if res["f1-score"] > max_f1:
                max_f1 = res["f1-score"]
                if group_number <= 5:
                    best_param = case[group_number]
                else:
                    best_param = (case[6], case[7])
            plotManager.add_plot(res, plot_type="F1-score")
        best_params_list.append(best_param)
        plotManager.save(str(group_number) + '-' + imname + ".png")
        group_number += 1

    best_params_dict = None
    if "word2vec".lower() in models_path.lower():
        best_params_dict = dict(zip(PARAMS_SEQUENCE_WORD2VEC, best_params_list))
    elif "fastText".lower() in models_path.lower():
        best_params_dict = dict(zip(PARAMS_SEQUENCE_FASTTEXT, best_params_list))

    with open(os.path.join(best_params_path, 'best_' + model_name + '.json'), "w") as outfile:
        json.dump(best_params_dict, outfile)


