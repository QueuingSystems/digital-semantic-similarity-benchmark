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


## Использование
Запуск: `python -m dss_benchmark`
