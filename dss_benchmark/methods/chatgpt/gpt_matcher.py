import hashlib
import json
import re
import time
import traceback
from dataclasses import dataclass, field
from pprint import pprint

import backoff
import cachetools
import openai
import tiktoken
from dss_benchmark.common import limit_array
from dss_benchmark.methods import AbstractSimilarityMethod

__all__ = ["GPTMatcher", "GPTMatcherParams"]


@dataclass
class GPTMatcherParams:
    max_tokens: int = field(
        default=4010,
        metadata={"help": "Максимальный размер промпта для GPT-3"},
    )
    temperature: float = field(default=0.75, metadata={"help": "Параметр температуры"})
    chat_model: str = field(default="gpt-3.5-turbo", metadata={"help": "OpenAI модель"})
    openai_key: str = field(default="", metadata={"help": "OpenAI API ключ"})


def _backoff_hdlr(details):
    traceback.print_exc()
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with args {args} and kwargs "
        "{kwargs}".format(**details)
    )
    time.sleep(5)


class GPTMatcher(AbstractSimilarityMethod):
    INIT_PROMPT = "You compare two texts, such as vacancies and resumes, and return integers from 0 to 10, where 0 means texts are not similar, and 10 means texts are similar"
    QUERY_PROMPT = "Compare the following two texts. Return integer from 0 to 10 and nothing else, where 0 means texts have nothing in common, and 10 means texts are very similar."

    def __init__(
        self, params: GPTMatcherParams, verbose=False, cache: cachetools.Cache = None
    ):
        self.params = params
        self.verbose = verbose
        self.cache = cache
        if len(params.openai_key) > 0:
            openai.api_key = params.openai_key
        self.encoding = tiktoken.get_encoding("cl100k_base")

        self.messages = []
        self.reset_messages()
        self.used_tokens = 0

    def reset_messages(self):
        self.messages = [
            {
                "role": "system",
                "content": self.INIT_PROMPT,
            }
        ]
        self.used_tokens = len(self.encoding.encode(self.INIT_PROMPT)) + 4

    def _crop_text_by_tokens(self, text: str, max_tokens: int):
        tokens = 0
        res = []
        for tok in text.split():
            tok_tokens = len(self.encoding.encode(tok))
            if tokens + tok_tokens > max_tokens:
                return " ".join(res)
            tokens += tok_tokens
            res.append(tok)
        return " ".join(res)

    def _make_prompt(self, text_1: str, text_2: str, max_tokens=3800):
        seed = f"{self.QUERY_PROMPT}\n[[Text 1]]: \n[[Text 2]]: \n[[Similarity from 0 to 10]]:"
        seed_tokens = len(self.encoding.encode(seed))
        max_tokens -= seed_tokens + 4
        text_1_tokens = len(self.encoding.encode(text_1))
        text_2_tokens = len(self.encoding.encode(text_2))
        if text_1_tokens + text_2_tokens > max_tokens:
            [text_1_tokens_c, text_2_tokens_c] = limit_array(
                [text_1_tokens, text_2_tokens], max_tokens
            )
            if text_1_tokens_c < text_1_tokens:
                text_1 = self._crop_text_by_tokens(text_1, text_1_tokens_c)
            if text_2_tokens_c < text_2_tokens:
                text_2 = self._crop_text_by_tokens(text_2, text_2_tokens_c)

        prompt = f"{self.QUERY_PROMPT}\n[[Text 1]]: {text_1}\n[[Text 2]]: {text_2}\n[[Similarity from 0 to 10]]:"
        return prompt

    def _get_hash_key(self, messages):
        messages_hash = hashlib.md5(json.dumps(messages).encode("utf-8")).hexdigest()
        values = [
            "openai:",
            self.params.max_tokens,
            self.params.chat_model,
            self.params.temperature,
            messages_hash,
        ]
        return "--".join([str(v) for v in values])

    @backoff.on_exception(
        backoff.expo,
        (
            openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.InvalidRequestError,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
        ),
        max_tries=3,
        on_backoff=_backoff_hdlr,
    )
    def _get_result(self, messages):
        if self.verbose:
            print("Prompting:")
            pprint(messages)
        key = self._get_hash_key(messages)
        cached = self.cache.get(key)
        if cached is not None:
            result = cached
            if self.verbose:
                print(f"Found cached: ", result)
        else:
            response = openai.ChatCompletion.create(
                model=self.params.chat_model,
                messages=messages,
                temperature=self.params.temperature,
                max_tokens=64,
            )
            result = response["choices"][0]["message"]["content"]
            if self.verbose:
                print(f"Received: {result}")
            try:
                result = int(result)
                self.cache[key] = result
            except ValueError:
                number = re.search(r"\d+", result).group()
                try:
                    result = int(number)
                    self.cache[key] = result
                except ValueError:
                    print(f"Bad result: {result}")
                    result = 0
        return result

    def add_shot(self, text_1: str, text_2: str, res: int, tokens=1000):
        prompt = self._make_prompt(text_1, text_2, tokens - 12)
        self.used_tokens += len(self.encoding.encode(prompt)) + 12
        self.messages.extend(
            [
                {"role": "user", "content": self._make_prompt(text_1, text_2)},
                {"role": "assistant", "content": str(res)},
            ]
        )

    def match(self, text_1: str, text_2: str) -> float:
        messages = [
            *self.messages,
            {
                "role": "user",
                "content": self._make_prompt(
                    text_1, text_2, self.params.max_tokens - self.used_tokens
                ),
            },
        ]
        return self._get_result(messages) * 10
