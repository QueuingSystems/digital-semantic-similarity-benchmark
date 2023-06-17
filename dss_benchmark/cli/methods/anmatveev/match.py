import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.common import parse_arbitrary_arguments
import pandas as pd
from dss_benchmark.methods.anmatveev.match.match import (
    MatchManager
)

image_path = "images/"

@click.group(
    "match", help="Сопоставление и исследование"
)

def mch():
    pass

@mch.command(
    help="Максимизировать F1-score с подбором оптимального cutoff", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mp", "--model_path", required=True, type=str, help="Путь к модели", prompt=True)
@click.option("-t", "--texts", required=True, type=str, help="Путь к бенчмарку", prompt=True)
def max_f1(model_path, texts):
    manager = MatchManager()
    manager.set_model(model_path)
    benchmark = pd.read_json(texts)
    preprocessed = manager.preprocess_and_save_pairs(benchmark, 'text_rp', 'text_proj')
    res = manager.max_f1(preprocessed['preprocessed_text_rp'],
                         preprocessed['preprocessed_text_proj'],
                         preprocessed,
                         model_path)
    print(res)