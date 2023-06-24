# Эксперименты по семантическому сопоставлению текстов
## Установка

Создание окружения через [conda](https://docs.conda.io/en/latest/miniconda.html):
```
conda env create -f environment.yml
```

После чего: `conda activate dssm`.

И настроить зависимости:
```
python -m spacy download ru_core_news_sm
python -m nltk.downloader stopwords
```

Для `KeywordDistanceMatcher` реализовано кэширование промежуточных результатов. Можно запустить Redis с паролем `12345` (параметры подключения - `dss_benchmark.common.cache.init_cache`). Иначе делается fallback на `cachetools.LRUCache`.

## Структура
- `dss_benchmark.methods` - тут лежат методы сопоставления текстов
  - `keyword_matching` - сопоставление с помощью извлечения ключевых слов (`KeywordDistanceMatcher`)
- `dss_benchmark.experiments` - тут лежат эксперименты
- `dss_benchmark.cli`- консольный интерфейс
  - `methods` - запуск методов на данной паре текстов
  - `experiments` - запуск экспериментов

Запуск CLI: `python -m dss_benchmark`.

Команды `keyword-matching-exp match`, `keyword-matching match` принимают на вход параметры модели. Параметры можно посмотреть по `keyword-matching params`.

`python -m dss_benchmark research  train-cascade -m word2vec -fp data/word2vec-cases.csv -tr 
data/documents_preprocessed.json -bt data/all_examples.json -t1 text_rp -t2 text_proj -mp models/word2vec --best_params_path data` - Каскадное обучение модели word2vec по файлу со сценарием по пути `data/word2vec-cases.csv`, где в качестве обучающего набора выступает `data/documents_preprocessed.json`, а в качестве бенчмарка `data/all_examples.json`. Сравниваются поля `text_rp` и `text_proj`, иерархия обученных моделей будет находитсья в `models/word2vec` 
конфигурация модели с наилучшими параметрами будет записана в папку `data`

`python -m dss_benchmark research  train-cascade -m fastText -fp data/fastText-cases.csv -tr 
data/documents_preprocessed.json -bt data/all_examples.json -t1 text_rp -t2 text_proj -mp models/fastText --best_params_path data` то же, но с FastText

`python -m dss_benchmark match  max-f1 -mp  models/word2vec/5/5-word2vec-1-30-7-5-100  -t data/studentor_partner.csv  -t1 text_1 -t2 text_2 -imp best_roc_auc_all_examples_studentor_partner_` максимизация f1-score на наилучшей модели word2vec по ROC-AUC, обученной на датасете documents.json на бенчмарке studentor_partner.csv

`python -m dss_benchmark match  max-f1 -mp  models/word2vec/5/5-word2vec-1-30-7-5-100  -t data/dataset_v6_r30.csv  -t1 resume -t2 vacancy -imp best_roc_auc_all_examples_dataset_v6_r30_`  Максимизация F1 на наилучшей модели word2vec по ROC-AUC, обученной на датасете documents.json на бенчмарке dataset_v6_r30.csv

-imp - префикс названия картинки, сделан для того, чтобы можно было маркировать графики, т.к. часть названия определяется автоматически.






