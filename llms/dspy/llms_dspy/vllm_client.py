import functools
import os
import random
import requests
import os
from dsp.modules.lm import LM
import subprocess
import re
import shutil
import time
import logging


class OpenAIVLLMClient(LM):
    def __init__(self, model, port, url="http://localhost", endpoint="v1/completions", **kwargs):
        super().__init__(model=model)
        self.endpoint = endpoint
        self.url = f"{url}:{port}"
        self.headers = {"Content-Type": "application/json"}

    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        logging.info(f"generation kwargs: {kwargs}")

        payload = {
            "model": kwargs["model"],
            "prompt": prompt,
            "max_tokens": kwargs["max_tokens"],
            "temperature": kwargs["temperature"],
        }

        response = requests.post(
            f"{self.url}/{self.endpoint}",
            json=payload,
            headers=self.headers,
        )

        try:
            json_response = response.json()
            completions = json_response["choices"]
            response = {
                "prompt": prompt,
                "choices": [{"text": c["text"]} for c in completions],
            }
            return response

        except Exception as e:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True

        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response


class VLLMClient(LM):
    def __init__(self, model, port, url="http://localhost", endpoint="generate", **kwargs):
        super().__init__(model=model)
        self.endpoint = endpoint
        self.url = f"{url}:{port}"
        self.headers = {"Content-Type": "application/json"}

    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        logging.info(f"generation kwargs: {kwargs}")

        payload = {
            "prompt": prompt,
            "sampling_params": {
                "max_tokens": kwargs["max_tokens"],
                "temperature": kwargs["temperature"],
            }
        }

        response = requests.post(
            f"{self.url}/{self.endpoint}",
            json=payload,
            headers=self.headers,
        )

        try:
            json_response = response.json()
            return json_response

        except Exception as e:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True

        response = self.request(prompt, **kwargs)
        return [r.replace(prompt, "") for r in response["text"]]

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response
