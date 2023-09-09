import os
import logging
import time
from enum import Enum, unique

import openai
from openai.error import RateLimitError

# import tiktoken

openai.api_key = "08b49556c7df46c9815c377eab0149a1"
openai.api_base =  "https://pvg-azure-openai-east-us.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = "2023-03-15-preview"

MAX_RETRIES = 20
RETRY_DELAY = 2


@unique
class ChatGPTModel(Enum):
    GPT_35_TURBO = 'gpt-35-turbo-0301'
    GPT_4_8K = 'gpt-4-8k-0314'
    GPT_4_32K = 'gpt-4-32k-0314'


@unique
class EmbeddingModel(Enum):
    TEXT_EMBEDDING_ADA_002_V2 = 'text-embedding-ada-002-v2'
    TEXT_EMBEDDING_ADA_002_V1 = 'text-embedding-ada-002-v1'


class ChatGPT:

    def __init__(self, **params):
        # parameters
        self.engine = params.get("engine", ChatGPTModel.GPT_35_TURBO.value)
        self.temperature = params.get("temperature", 0.5)
        self.max_tokens = params.get("maxTokens", 800)
        self.top_p = params.get("topP", 0.95)
        self.start = params.get("start", ["<|im_start|>"])
        self.stop = params.get("stop", ["<|im_end|>"])

    def _create_prompt(self, system_message: str, messages: list) -> str:
        start = self.start[0]
        stop = self.stop[0]
        prompt = f"{start}system\n{system_message}\n{stop}"
        for message in messages:
            prompt += f"\n{start}{message['user']}\n{message['text']}\n{stop}"
        prompt += f"\n{start}bot\n"
        return prompt

    def _create_chat_prompt(self, system_message: str, messages: list) -> str:
        prompt = [{"role": "system", "content": system_message}]
        prompt.extend([{
            "role": "assistant" if x['user'] == 'bot' else "user",
            "content": x['text'],
        } for x in messages])

        return prompt

    def completion(self, data: dict) -> dict:
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                response = openai.Completion.create(
                    engine=ChatGPTModel.GPT_35_TURBO.value,
                    prompt=self._create_prompt(data["system"], data["messages"]),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=self.stop,
                )
                break
            except RateLimitError:
                logging.warning(f"API rate limit reached. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                retry_count += 1

        return response

    def chat_completion(self, data: dict) -> dict:
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                response = openai.ChatCompletion.create(
                    engine=self.engine,
                    messages=self._create_chat_prompt(data["system"], data["messages"]),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                break
            except RateLimitError:
                logging.warning(f"API rate limit reached. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                retry_count += 1

        return response


class Embedding:

    def __init__(self, **params):
        # parameters
        self.engine: str = params.get(
            "engine",
            EmbeddingModel.TEXT_EMBEDDING_ADA_002_V2.value,
        )
        self.result_: list = []

    def run(self, text: str) -> "Embedding":
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                self.result_ = openai.Embedding.create(
                    deployment_id=self.engine,
                    input=text,
                )["data"][0]["embedding"]
                break
            except RateLimitError:
                logging.warning(f"API rate limit reached. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                retry_count += 1

        return self
