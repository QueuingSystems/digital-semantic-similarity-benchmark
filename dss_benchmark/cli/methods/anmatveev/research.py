import os

import click
from dss_benchmark.methods.anmatveev.train import *
from dss_benchmark.methods.anmatveev.match import *
from dss_benchmark.methods.anmatveev.plot import *
from dss_benchmark.methods.anmatveev.params_parser import *
import pandas as pd

PARAMS_SEQUENCE_WORD2VEC = ["window", "epochs", "sg", "min_count", "vector_size"]
PARAMS_SEQUENCE_FASTTEXT = PARAMS_SEQUENCE_WORD2VEC[:] + ["min_n-max_n"]


@click.group(
    "research", help="Сопоставление и исследование"
)
def rsch():
    pass


def update_params(res_: dict, params_: TrainModelParams, model_: str):
    res_['sg'] = params_.sg
    res_['window'] = params_.window
    res_['epochs'] = params_.epochs
    res_['min_count'] = params_.min_count
    res_['vector_size'] = params_.vector_size
    if model_.lower() == "fasttext":
        res_['min_n'] = params_.min_n
        res_['max_n'] = params_.max_n


@rsch.command(
    help="Обучить все возможные комбинации моделей",
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("-m", "--model", required=True, type=str, help="Тип модели", prompt=True)
@click.option("-fp", "--file_path", required=True, type=str, help="Путь к файлу с параметрами", prompt=True)
@click.option("-tr", "--train_text", required=True, type=str, help="Обучающий набор", prompt=True)
@click.option("-bt", "--benchmark_text", required=True, type=str, help="Текст бенчмарка", prompt=True)
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.option("-mp", "--models_path", required=True, type=str, help="Путь к моделям", prompt=True)
@click.option("-bpp", "--best_params_path", required=True, type=str, help="Путь к папке, в которой будет лежать файл с оптимальными параметрами",
              prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
# pretrain models before its comparing with splitting into groups
def train_cascade(model, file_path, train_text, benchmark_text, text1, text2, models_path, best_params_path, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    params.texts = train_text
    os.makedirs(models_path, exist_ok=True)
    scenario = pd.read_csv(file_path)
    best_params = dict.fromkeys(list(scenario.columns)[1:])
    max_auc = 0
    if '.json' in benchmark_text:
        benchmark_text = pd.read_json(benchmark_text)
    elif '.csv' in benchmark_text:
        benchmark_text = pd.read_csv(benchmark_text)
    new_group = False
    current_group = 0
    plotManager = None
    imname = None
    for index, row in scenario.iterrows():

        if row['group'] != current_group:
            current_group = row['group']
            new_group = True
        else:
            new_group = False

        subdir = os.path.join(models_path, str(row['group']))
        os.makedirs(subdir, exist_ok=True)
        params.sg = int(row['sg']) if row['sg'] != 'x' else best_params['sg']
        params.window = int(row['window']) if row['window'] != 'x' else best_params['window']
        params.epochs = int(row['epochs']) if row['epochs'] != 'x' else best_params['epochs']
        params.min_count = int(row['min_count']) if row['min_count'] != 'x' else best_params['min_count']
        params.vector_size = int(row['vector_size']) if row['vector_size'] != 'x' else best_params['vector_size']
        model_name = str(row['group']) + '-' + model + '-' + str(params.sg) + \
                     '-' + str(params.window) + '-' + str(params.epochs) + '-' + str(params.min_count) + \
                     '-' + str(params.vector_size)
        if model.lower() == "fasttext":
            params.min_n = int(row['min_n']) if row['min_n'] != 'x' else best_params['min_n']
            params.max_n = int(row['max_n']) if row['max_n'] != 'x' else best_params['max_n']
            model_name += '-' + str(params.min_n) + '-' + str(params.max_n)
        trainManager = TrainModelManager(params)
        new_model_path = os.path.join(subdir, model_name)

        if not os.path.exists(new_model_path):
            trainManager.train(model, model_path=new_model_path)

        matchManager = MatchManager(new_model_path, params)
        res = matchManager.roc_auc(benchmark_text[text1],
                                   benchmark_text[text2],
                                   benchmark_text,
                                   new_model_path)

        update_params(res, params, model)
        if res["auc"] > max_auc:
            print("AUC = {}, cutoff = {}".format(res["auc"], res["cutoff"]))
            max_auc = res["auc"]
            update_params(best_params, params, model)

        print(f'Best params: {best_params}')

        if new_group:
            if plotManager:
                plotManager.save(imname + ".png")
            plotManager = PlotManager()
            imname = f"ROC-AUC-{row['group']}-{model}"
            print(new_group, imname)
            plotManager.init_plot(title=imname,
                                  xlabel="False Positive Rate",
                                  ylabel="True Positive Rate",
                                  model=model,
                                  plot_type="ROC-AUC",
                                  figsize=(7, 6))
        plotManager.add_plot(res)
    plotManager.save(imname + ".png")
    with open(os.path.join(best_params_path, 'best_roc_auc_' + model + '.json'), "w") as outfile:
        json.dump(best_params, outfile)



@rsch.command(
    help="Подбор оптимальных параметров для максимизации F1-score",
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mp", "--models_path", required=True, type=str, help="Путь к папке с моделями", prompt=True)
@click.option("-bt", "--benchmark_text", required=True, type=str, help="Путь к бенчмарку", prompt=True)
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.option("-bpp", "--best_params_path", required=True, type=str, help="Путь к файлу с оптимальными параметрами",
              prompt=True)
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
                              plot_type="F1-score",
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
            plotManager.add_plot(res)
        best_params_list.append(best_param)
        plotManager.save(str(group_number) + '-' + imname + ".png")
        group_number += 1

    best_params_dict = None
    if "word2vec".lower() in models_path.lower():
        best_params_dict = dict(zip(PARAMS_SEQUENCE_WORD2VEC, best_params_list))
    elif "fastText".lower() in models_path.lower():
        best_params_dict = dict(zip(PARAMS_SEQUENCE_FASTTEXT, best_params_list))

    with open(os.path.join(best_params_path, 'best_f1_score_' + model_name + '.json'), "w") as outfile:
        json.dump(best_params_dict, outfile)


@rsch.command(
    help="Подбор оптимальных параметров для максимизации F1-score",
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mp", "--models_path", required=True, type=str, help="Путь к папке с моделями", prompt=True)
@click.option("-bt", "--benchmark_text", required=True, type=str, help="Путь к бенчмарку", prompt=True)
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.option("-bpp", "--best_params_path", required=True, type=str, help="Путь к файлу с оптимальными параметрами",
              prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
# cases in a file will be splitted into groups with a space
def get_best_params_roc_auc(models_path, benchmark_text, text1, text2, best_params_path, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    benchmark_text = pd.read_json(benchmark_text)
    best_params_list = []
    group_number = 1
    model_name = models_path.split('/')[-1]

    for group, dirs, files in sorted(os.walk(models_path))[1:]:
        plotManager = PlotManager()
        imname = f"ROC-AUC-{model_name}"
        plotManager.init_plot(title=imname,
                              xlabel="False Positive Rate",
                              ylabel="True Positive Rate",
                              model=model_name,
                              plot_type="ROC-AUC",
                              figsize=(7, 6))
        max_auc = 0
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
            res = matchManager.roc_auc(benchmark_text[text1], benchmark_text[text2], benchmark_text,
                                       model_path)

            res["window"] = params.window
            res["epochs"] = params.epochs
            res["sg"] = params.sg
            res["min_count"] = params.min_count
            res["vector_size"] = params.vector_size
            res["min_n-max_n"] = (params.min_n, params.max_n)
            if res["auc"] > max_auc:
                max_auc = res["auc"]
                if group_number <= 5:
                    best_param = case[group_number]
                else:
                    best_param = (case[6], case[7])
            plotManager.add_plot(res)
        best_params_list.append(best_param)
        plotManager.save(str(group_number) + '-' + imname + ".png")
        group_number += 1

    best_params_dict = None
    if "word2vec".lower() in models_path.lower():
        best_params_dict = dict(zip(PARAMS_SEQUENCE_WORD2VEC, best_params_list))
    elif "fastText".lower() in models_path.lower():
        best_params_dict = dict(zip(PARAMS_SEQUENCE_FASTTEXT, best_params_list))

    with open(os.path.join(best_params_path, 'best_roc_auc_' + model_name + '.json'), "w") as outfile:
        json.dump(best_params_dict, outfile)
