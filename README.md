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
