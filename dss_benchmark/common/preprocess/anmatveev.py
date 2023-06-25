import re

import pandas as pd
import pymorphy2
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

__all__ = ["preprocess", "preprocess_and_save", "sent_preprocess"]

PUNCTUATION_MARKS = [
    "!",
    ",",
    "(",
    ")",
    ";",
    ":",
    "-",
    "?",
    ".",
    "..",
    "...",
    '"',
    "/",
    "\`\`",
    "»",
    "«",
]
STOP_WORDS = stopwords.words("russian")
MORPH = pymorphy2.MorphAnalyzer()


def preprocess(text: str):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in PUNCTUATION_MARKS:
            lemma = MORPH.parse(token)[0].normal_form
            if re.match(r"(\d.|\d)", lemma) is None:
                if lemma not in STOP_WORDS:
                    preprocessed_text.append(lemma)
    return preprocessed_text


def preprocess_and_save(data_df: pd.DataFrame, path, text_field="text") -> pd.DataFrame:
    # for preprocessing dataset. Use it only in critical cases cause it's too slow on big datasets
    data_df["preprocessed_" + text_field] = data_df.apply(
        lambda row: preprocess(row[text_field], PUNCTUATION_MARKS, STOP_WORDS, MORPH),
        axis=1,
    )
    data_df_preprocessed = data_df.copy()
    data_df_preprocessed = data_df_preprocessed.drop(columns=[text_field], axis=1)
    data_df_preprocessed.reset_index(drop=True, inplace=True)
    if path is not None:
        data_df_preprocessed.to_json(path)
    return data_df_preprocessed


def sent_preprocess(text: str):
    preprocessed_text = sent_tokenize(text)
    for i in range(len(preprocessed_text)):
        res = re.sub(r"([^\w\s])|([0-9]+)", "", preprocessed_text[i])
        # res = re.sub(r'', '' ,  res)
        preprocessed_text[i] = res
    preprocessed_text = list(filter(lambda sentence: sentence != "", preprocessed_text))
    return preprocessed_text
