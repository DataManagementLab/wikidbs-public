########################################################################################################################
# OpenAI API helpers version: 2024-09-29
#
# use the following methods:
# openai_model(...)        ==> get model info
# openai_execute(...)      ==> execute API requests
# openai_cost_for_cache()  ==> compute total cost of all cached responses
#
# You must store your OpenAI API key in an environment variable, for example using:
# export OPENAI_API_KEY="<your-key>"
#
# To call openai_execute(...) from multiple processes, you must use a global context:
# with multiprocessing.Manager() as manager:
#     context = manager.dict()
#     semaphore = manager.Semaphore()
#     # every call now requires the context and semaphore:
#     responses = openai_execute(requests, global_context=context, global_semaphore=semaphore)
########################################################################################################################

import dataclasses
import functools
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
from typing import Literal, Any

import requests
import tiktoken
import tqdm

logger = logging.getLogger(__name__)

CACHE_PATH = pathlib.Path("data/openai_cache")

MODEL_PARAMETERS = {  # see https://platform.openai.com/docs/models and https://openai.com/pricing
    # GPT-3.5 Turbo Instruct
    "gpt-3.5-turbo-instruct-0914": {
        "chat_or_completion": "completion",
        "cost_per_1k_input_tokens": 0.0015,  # taken from "gpt-4"
        "cost_per_1k_output_tokens": 0.0020,  # taken from "gpt-4"
        "max_context": 4_096,
        "max_output_tokens": 4_096
    },

    # GPT-3.5 Turbo
    "gpt-3.5-turbo-1106": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.0010,
        "cost_per_1k_output_tokens": 0.0020,
        "max_context": 16_385,
        "max_output_tokens": 4_096
    },
    "gpt-3.5-turbo-0125": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.0005,
        "cost_per_1k_output_tokens": 0.0015,
        "max_context": 16_385,
        "max_output_tokens": 4_096
    },

    # GPT-4 Turbo and GPT-4
    "gpt-4-0314": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.0300,  # taken from "gpt-4"
        "cost_per_1k_output_tokens": 0.0600,  # taken from "gpt-4"
        "max_context": 8_192,
        "max_output_tokens": 8_192
    },
    "gpt-4-0613": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.0300,  # taken from "gpt-4"
        "cost_per_1k_output_tokens": 0.0600,  # taken from "gpt-4"
        "max_context": 8_192,
        "max_output_tokens": 8_192
    },
    "gpt-4-1106-preview": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.0100,
        "cost_per_1k_output_tokens": 0.0300,
        "max_context": 128_000,
        "max_output_tokens": 4_096
    },
    "gpt-4-0125-preview": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.0100,
        "cost_per_1k_output_tokens": 0.0300,
        "max_context": 128_000,
        "max_output_tokens": 4_096
    },
    "gpt-4-turbo-2024-04-09": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.0100,
        "cost_per_1k_output_tokens": 0.0300,
        "max_context": 128_000,
        "max_output_tokens": 4_096
    },

    # GPT-4o and GPT-4o mini
    "gpt-4o-2024-05-13": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.00500,
        "cost_per_1k_output_tokens": 0.01500,
        "max_context": 128_000,
        "max_output_tokens": 4_096
    },
    "gpt-4o-2024-08-06": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.00250,
        "cost_per_1k_output_tokens": 0.01000,
        "max_context": 128_000,
        "max_output_tokens": 16_384
    },
    "gpt-4o-mini-2024-07-18": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.000150,
        "cost_per_1k_output_tokens": 0.000600,
        "max_context": 128_000,
        "max_output_tokens": 16_384
    },

    # o1 preview and o1 mini
    "o1-preview-2024-09-12": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.015,
        "cost_per_1k_output_tokens": 0.060,
        "max_context": 128_000,
        "max_output_tokens": 32_768
    },
    "o1-mini-2024-09-12": {
        "chat_or_completion": "chat",
        "cost_per_1k_input_tokens": 0.003,
        "cost_per_1k_output_tokens": 0.012,
        "max_context": 128_000,
        "max_output_tokens": 65_536
    }
}


########################################################################################################################
# API
########################################################################################################################


def openai_model(
        model: str
) -> dict:
    """Get information about the OpenAI model.

    Args:
        model: The name of the model.

    Returns:
        A dictionary with information about the OpenAI model.
    """
    return _get_model_params(model)


def openai_execute(
        requests: list[dict],
        *,
        force: float | None = None,
        silent: bool = False,
        global_context: dict | None = None,
        global_semaphore: "multiprocessing.Semaphore | None" = None
) -> list[dict]:
    """Execute a list of requests against the OpenAI API.

    This method also computes the maximum cost incurred by the requests, caches requests and responses, and waits
    between requests to abide the API limits.

    Args:
        requests: A list of API requests.
        force: An optional float specifying the cost below or equal to which no confirmation should be required.
        silent: Whether to display log messages and progress bars.
        global_context: Optional global context for use with multiprocessing.
        global_semaphore: Optional global semaphore for use with multiprocessing.

    Returns:
        A list of API responses.
    """
    global _local_context, _local_semaphore

    if (global_context is None) != (global_semaphore is None):
        raise AssertionError("You must provide either both `global_context` and `global_semaphore` or neither!")

    if global_context is not None:
        context = global_context
        semaphore = global_semaphore
    else:
        context = _local_context
        semaphore = _local_semaphore

    with semaphore:
        if "num_running" not in context.keys():
            context["num_running"] = 0

    pairs = [_Pair(_Request(request)) for request in requests]

    with _ProgressBar(total=len(pairs), desc="", disable=silent) as progress_bar:

        # check requests
        progress_bar.set_description("check requests")
        progress_bar.reset(total=len(pairs))
        for pair in pairs:
            pair.request.check()
            progress_bar.update()

        # create cache directory
        CACHE_PATH.mkdir(parents=True, exist_ok=True)

        # load cached pairs
        pairs_to_execute = []
        progress_bar.set_description("load responses")
        progress_bar.reset(total=len(pairs))
        for pair in pairs:
            pair.response = pair.request.load_cached_response()
            if pair.response is None:
                pairs_to_execute.append(pair)
            else:
                progress_bar.cached += 1
            progress_bar.update()

        # in case some pairs were not cached, execute them
        if len(pairs_to_execute) > 0:
            progress_bar.clear()  # clear before printing/logging

            if "OPENAI_API_KEY" not in os.environ.keys():
                raise AssertionError(f"Missing `OPENAI_API_KEY` in environment variables!")

            # compute maximum cost
            total_max_cost = sum(pair.request.max_cost() for pair in pairs_to_execute)
            if force is None or total_max_cost > force:
                logger.info(f"press enter to continue and spend up to around ${total_max_cost:.2f}")
                input(f"press enter to continue and spend up to around ${total_max_cost:.2f}")
            elif not silent and total_max_cost > 0:
                logger.info(f"spending up to around ${total_max_cost:.2f}")

            # sort requests to execute longest requests first, put one short request first to quickly obtain HTTP header
            pairs_to_execute.sort(key=lambda p: p.request.max_total_usage(), reverse=True)
            pairs_to_execute = pairs_to_execute[-1:] + pairs_to_execute[:-1]

            # execute requests
            progress_bar.set_description("execute requests")
            progress_bar.reset(total=len(pairs))
            progress_bar.update(progress_bar.cached)
            while True:  # break if all(pair.status == "done" for pair in pairs_to_execute)
                logger.debug("repeatedly iterate through pairs until all are done")
                for pair in pairs_to_execute:
                    with semaphore:
                        if pair.status != "open":
                            continue
                        pair.status = "waiting"

                        if pair.request.model not in context.keys():
                            context[pair.request.model] = _ModelBudgetState.new()

                    while True:  # break if request is "done" in sequential execution or "running" in parallel execution
                        with semaphore:
                            context[pair.request.model] = context[pair.request.model].consider_time()

                            if context[pair.request.model].is_enough_for_request(pair.request):
                                match context[pair.request.model].mode:
                                    case "sequential" if context["num_running"] == 0:
                                        logger.debug(f"sequential execution for `{pair.request.model}`: execute")
                                        progress_bar.bottleneck = "P"

                                        pair.status = "running"
                                        context["num_running"] = context["num_running"] + 1
                                        progress_bar.running = context["num_running"]
                                        progress_bar.update_postfix()

                                        context[pair.request.model] = context[pair.request.model].decrease_by_request(
                                            pair.request
                                        )

                                        http_response = pair.request.execute()
                                        pair.response = _Response(http_response.json())

                                        context[pair.request.model] = context[pair.request.model].set_from_headers(
                                            http_response.headers
                                        )

                                        context["num_running"] = context["num_running"] - 1
                                        progress_bar.running = context["num_running"]
                                        progress_bar.cost += pair.response.total_cost()

                                        match http_response.status_code:
                                            case 200:
                                                context[pair.request.model] = context[pair.request.model].to_parallel()
                                                pair.status = "done"
                                                progress_bar.update()
                                                break
                                            case 429:
                                                pair.status = "open"
                                                progress_bar.update_postfix()  # not done -> update only postfix
                                            case _:
                                                pair.status = "done"
                                                progress_bar.failed += 1
                                                progress_bar.update()
                                                break
                                    case "parallel" if context["num_running"] < 200:  # max. num. of parallel requests
                                        logger.debug(f"parallel execution for `{pair.request.model}`: execute")
                                        progress_bar.bottleneck = "P"

                                        pair.status = "running"
                                        context["num_running"] = context["num_running"] + 1
                                        progress_bar.running = context["num_running"]
                                        progress_bar.update_postfix()

                                        context[pair.request.model] = context[pair.request.model].decrease_by_request(
                                            pair.request
                                        )

                                        def execute(p: _Pair, pb: _ProgressBar, c: dict,
                                                    s: threading.Semaphore) -> None:
                                            http_response = p.request.execute()
                                            p.response = _Response(http_response.json())

                                            with s:
                                                c[pair.request.model] = c[p.request.model].set_from_headers(
                                                    http_response.headers
                                                )

                                                c["num_running"] = c["num_running"] - 1
                                                pb.running = c["num_running"]
                                                pb.cost += p.response.total_cost()

                                                match http_response.status_code:
                                                    case 200:
                                                        c[p.request.model] = c[p.request.model].increase_by_response(
                                                            p.request,
                                                            p.response
                                                        )
                                                        p.status = "done"
                                                        pb.update()
                                                    case 429:
                                                        logger.debug(
                                                            f"parallel execution for `{p.request.model}`: "
                                                            f"rate limit error -> switch to sequential execution"
                                                        )
                                                        p.status = "open"
                                                        c[p.request.model] = c[p.request.model].to_sequential()
                                                        pb.update_postfix()  # not done -> update only postfix
                                                    case _:
                                                        p.status = "done"
                                                        pb.failed += 1
                                                        pb.update()

                                        pair.thread = threading.Thread(
                                            target=execute,
                                            args=(pair, progress_bar, context, semaphore)
                                        )
                                        pair.thread.start()
                                        break
                                    case _:
                                        progress_bar.bottleneck = "T"
                                        progress_bar.update_postfix()
                            else:
                                progress_bar.bottleneck = "L"
                                progress_bar.update_postfix()

                        time.sleep(0.05)  # sleep to wait for thread limit or rate limit budget

                if all(pair.status == "done" for pair in pairs_to_execute):
                    break
                progress_bar.bottleneck = "S"
                progress_bar.update_postfix()
                time.sleep(1)  # sleep to wait for stragglers and failures

            for pair in pairs_to_execute:
                if pair.thread is not None:
                    pair.thread.join()

    return [pair.response.response for pair in pairs]


def openai_cost_for_cache() -> float:
    """Compute the total dollar cost incurred by executing all cached requests/responses.

    Returns:
        The total dollar cost incurred by executing all cached requests/responses.
    """
    file_paths = list(sorted(CACHE_PATH.glob(f"*.json")))
    total_cost = 0
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            cached_pair = json.load(file)
            cached_response = _Response(cached_pair["response"])
            total_cost += cached_response.total_cost()

    return total_cost


########################################################################################################################
# implementation
########################################################################################################################


_local_context = {}
_local_semaphore = threading.Semaphore()


@functools.cache
def _get_model_params(model: str) -> dict:
    if model not in MODEL_PARAMETERS.keys():
        raise AssertionError(f"Unknown model `{model}`!")
    else:
        return MODEL_PARAMETERS[model]


@functools.cache
def _get_encoding_cached(model: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(model)


class _Request:
    request: dict

    def __init__(self, request: dict) -> None:
        self.request = request

    @functools.cached_property
    def model(self) -> str:
        if "model" not in self.request.keys():
            raise AttributeError("Missing field `model` in request!")
        return self.request["model"]

    @functools.cached_property
    def messages(self) -> list[dict]:
        if "messages" not in self.request.keys():
            raise AttributeError("Missing field `messages` in request, which is required for this model!")
        return self.request["messages"]

    @functools.cached_property
    def prompt(self) -> str:
        if "prompt" not in self.request.keys():
            raise AttributeError("Missing field `prompt` in request, which is required for this model!")
        return self.request["prompt"]

    @functools.cache
    def is_chat_or_completion(self) -> str:  # use `model` to determine if request is for chat or completion
        return _get_model_params(self.model)["chat_or_completion"]

    @functools.cache
    def url(self) -> str:
        match self.is_chat_or_completion():
            case "chat":
                return "https://api.openai.com/v1/chat/completions"
            case "completion":
                return "https://api.openai.com/v1/completions"
            case _:
                raise AssertionError(f"Invalid parameter `chat_or_completion` for model `{self.model}`!")

    @functools.cache
    def num_input_tokens(self) -> int:
        encoding = _get_encoding_cached(self.model)

        match self.is_chat_or_completion():
            case "chat":
                extra_tokens = 5  # number of additional tokens in each message
                return sum(len(encoding.encode(message["content"])) + extra_tokens for message in self.messages)
            case "completion":
                return len(encoding.encode(self.prompt))
            case _:
                raise AssertionError(f"Invalid parameter `chat_or_completion` for model `{self.model}`!")

    @functools.cache
    def max_num_output_tokens(self) -> int:
        if "max_completion_tokens" in self.request.keys() and self.request["max_completion_tokens"] is not None:
            return self.request["max_completion_tokens"]
        elif "max_tokens" in self.request.keys() and self.request["max_tokens"] is not None:
            return self.request["max_tokens"]
        else:
            model_params = _get_model_params(self.model)
            left_for_output = max(0, model_params["max_context"] - self.num_input_tokens())
            if model_params["max_output_tokens"] is not None and model_params["max_output_tokens"] < left_for_output:
                return model_params["max_output_tokens"]
            else:
                return left_for_output

    @functools.cache
    def max_total_tokens(self) -> int:
        return self.num_input_tokens() + self.max_num_output_tokens()

    @functools.cache
    def max_input_usage(self) -> int:
        if "best_of" in self.request.keys():
            n = self.request["best_of"]
        elif "n" in self.request.keys():
            n = self.request["n"]
        else:
            n = 1
        return n * self.num_input_tokens()

    @functools.cache
    def max_output_usage(self) -> int:
        if "best_of" in self.request.keys():
            n = self.request["best_of"]
        elif "n" in self.request.keys():
            n = self.request["n"]
        else:
            n = 1
        return n * self.max_num_output_tokens()

    @functools.cache
    def max_total_usage(self) -> int:
        return self.max_input_usage() + self.max_output_usage()

    @functools.cache
    def max_cost(self) -> float:
        model_params = _get_model_params(self.model)
        input_cost = self.max_input_usage() * (model_params["cost_per_1k_input_tokens"] / 1000)
        output_cost = self.max_output_usage() * (model_params["cost_per_1k_output_tokens"] / 1000)
        return input_cost + output_cost

    @functools.cache
    def hash(self) -> str:
        return hashlib.sha256(bytes(json.dumps(self.request), "utf-8")).hexdigest()

    def check(self) -> None:
        model_params = _get_model_params(self.model)

        if self.num_input_tokens() > model_params["max_context"]:
            logger.warning("request's number of input tokens exceeds model's `max_context`")

        if "max_tokens" in self.request.keys() and "max_completion_tokens" in self.request.keys():
            logger.warning("request contains `max_tokens` and `max_completion_tokens`")

        if "max_completion_tokens" in self.request.keys():
            token_limit = "max_tokens"
        elif "max_completion_tokens" in self.request.keys():
            token_limit = "max_completion_tokens"
        else:
            token_limit = None

        if token_limit is not None:
            if model_params["max_output_tokens"] is not None and token_limit > model_params["max_output_tokens"]:
                logger.warning("request's `max_tokens` or `max_completion_tokens` exceeds model's `max_output_tokens`")

            if self.num_input_tokens() + token_limit > model_params["max_context"]:
                logger.warning(
                    "request's input tokens + `max_tokens` or `max_completion_tokens` exceeds model's `max_context`")

        if "seed" not in self.request.keys():
            logger.warning("missing optional field `seed`, which is required for reproducibility")

        if "temperature" not in self.request.keys() or self.request["temperature"] != 0:
            logger.warning("request's `temperature` not set to 0, which is required for reproducibility")

    def load_cached_response(self):  # -> "_Response" | None
        path = CACHE_PATH / f"{self.hash()}.json"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as file:
                cached_pair = json.load(file)
            cached_request = _Request(cached_pair["request"])
            cached_response = _Response(cached_pair["response"])
            if self.request == cached_request.request:
                return cached_response
        return None

    def execute(self) -> requests.Response:
        http_response = requests.post(
            url=self.url(),
            json=self.request,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        )

        if http_response.status_code == 200:
            path = CACHE_PATH / f"{self.hash()}.json"
            with open(path, "w", encoding="utf-8") as cache_file:
                json.dump({"request": self.request, "response": http_response.json()}, cache_file)
        elif http_response.status_code == 429:
            logger.info("retry request due to rate limit error")
        else:
            logger.warning(f"request failed, no retry: {http_response.content}")

        return http_response


class _Response:
    response: dict

    def __init__(self, response: dict) -> None:
        self.response = response

    @functools.cache
    def was_successful(self) -> bool:  # use entry `choices` to determine if request was successful
        return "choices" in self.response.keys()

    @functools.cached_property
    def model(self) -> str:
        if "model" not in self.response.keys():
            raise AttributeError("Missing field `model` in response, which is required for successful requests!")
        return self.response["model"]

    @functools.cached_property
    def usage(self) -> dict:
        if "usage" not in self.response.keys():
            raise AttributeError("Missing field `usage` in response, which is required for successful requests!")
        return self.response["usage"]

    @functools.cache
    def total_usage(self) -> int:
        if self.was_successful():
            return self.usage["total_tokens"]
        else:
            return 0

    @functools.cache
    def total_cost(self) -> float:
        if self.was_successful():
            model_params = _get_model_params(self.model)
            total_cost = 0
            if "prompt_tokens" in self.usage.keys():
                total_cost += self.usage["prompt_tokens"] * (model_params["cost_per_1k_input_tokens"] / 1000)
            if "completion_tokens" in self.usage.keys():
                total_cost += self.usage["completion_tokens"] * (model_params["cost_per_1k_output_tokens"] / 1000)
            return total_cost
        else:
            return 0


@dataclasses.dataclass
class _Pair:
    request: _Request
    response: _Response | None = None
    status: Literal["open"] | Literal["waiting"] | Literal["running"] | Literal["done"] = "open"
    thread: threading.Thread | None = None


@dataclasses.dataclass
class _ModelBudgetState:
    mode: Literal["sequential"] | Literal["parallel"]
    rpm: int | None
    tpm: int | None
    r: int | None
    t: int | None
    last_update: float

    @classmethod
    def new(cls) -> "_ModelBudgetState":
        return cls("sequential", None, None, None, None, time.time())

    def is_enough_for_request(self, request: _Request) -> bool:
        return (self.r is None or self.r >= 1) and (self.t is None or self.t >= request.max_total_usage())

    def consider_time(self) -> "_ModelBudgetState":
        now = time.time()
        delta = now - self.last_update
        if self.rpm is not None and self.r is not None:
            self.r = min(self.rpm, int(self.r + self.rpm * delta / 60))
        if self.tpm is not None and self.t is not None:
            self.t = min(self.tpm, int(self.t + self.tpm * delta / 60))
        self.last_update = now
        return self

    def decrease_by_request(self, request: _Request) -> "_ModelBudgetState":
        if self.r is not None:
            self.r -= 1
        if self.t is not None:
            self.t -= request.max_total_usage()
        return self

    def increase_by_response(self, request: _Request, response: _Response) -> "_ModelBudgetState":
        if response.total_usage() < request.max_total_usage():
            self.t = min(self.tpm, int(self.t + request.max_total_usage() - response.total_usage()))
        return self

    def set_from_headers(self, headers: dict[str, Any]) -> "_ModelBudgetState":
        if "x-ratelimit-limit-requests" in headers.keys():
            self.rpm = int(headers["x-ratelimit-limit-requests"])
        if "x-ratelimit-limit-tokens" in headers.keys():
            self.tpm = int(headers["x-ratelimit-limit-tokens"])
        if "x-ratelimit-remaining-requests" in headers.keys():
            header_r = int(headers["x-ratelimit-remaining-requests"])
            if self.r is None or self.r > header_r:
                self.r = header_r
        if "x-ratelimit-remaining-tokens" in headers.keys():
            header_t = int(headers["x-ratelimit-remaining-tokens"])
            if self.t is None or self.t > header_t:
                self.t = header_t
        return self

    def to_parallel(self) -> "_ModelBudgetState":
        self.mode = "parallel"
        return self

    def to_sequential(self) -> "_ModelBudgetState":
        self.mode = "sequential"
        return self


class _ProgressBar(tqdm.tqdm):
    running: int
    failed: int
    cached: int
    cost: float
    bottleneck: Literal["T"] | Literal["L"] | Literal["P"] | Literal["S"]
    bottleneck_counter: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.running = 0
        self.failed = 0
        self.cached = 0
        self.cost = 0
        self.bottleneck = "P"
        self.update_postfix()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return super().__exit__(exc_type, exc_value, traceback)

    def update(self, *args, **kwargs) -> None:
        self.update_postfix()
        super().update(*args, **kwargs)

    def update_postfix(self) -> None:
        self.set_postfix_str(
            f"{self.bottleneck}{self.running:03d}, failed={self.failed}, cached={self.cached}, cost=${self.cost:.2f}"
        )
