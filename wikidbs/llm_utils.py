import collections
import logging
import pathlib
import json
import re
from copy import deepcopy

from wikidbs.openai import openai_execute


logger = logging.getLogger(__name__)

_openai_request_seed: int = 321164097


def fill_chat_template(
        template: list[dict[str, str] | str],
        **args
    ) -> list[dict[str, str]]:
    """Replace {{variables}} in the template with the given values.

    A variable can be a list of messages, a message, or a string.

    Warns in case of missing values, but not in case of unneeded values.

    Args:
        template: List of template messages containing {{variables}}.
        **args: The given string or message values for the variables.

    Returns:
        The filled-out template.
    >>> fill_chat_template([{"role": "user", "content": "My name is {{name}}."}, "{{greeting}}"], name="Micha", greeting={"role": "assistant", "content": "Nice to meet you!"})
    [{'role': 'user', 'content': 'My name is Micha.'}, {'role': 'assistant', 'content': 'Nice to meet you!'}]
    """
    template = deepcopy(template)
    # replace variables with values
    new_template = []
    for message in template:
        if isinstance(message, str):
            for key, value in args.items():
                template_key = "{{" + key + "}}"
                if template_key == message:
                    if isinstance(value, list):
                        new_template += value
                    elif isinstance(value, dict):
                        new_template.append(value)
                    else:
                        raise TypeError(f"Value for key '{key}' must be a message dictionary or list of message dictionaries!")
                    break
            else:
                raise AssertionError(f"Missing value for template message variable '{message}'!")
        elif isinstance(message, dict):
            for key, value in args.items():
                template_key = "{{" + key + "}}"
                if template_key in message["content"]:
                    message["content"] = message["content"].replace(template_key, value)
            new_template.append(message)
        else:
            raise TypeError(f"Invalid type {type(message)} for template message!")

    # check that all variables have been filled
    for message in new_template:
        # noinspection RegExpRedundantEscape
        missing_keys = re.findall(r"\{\{(.*)\}\}", message["content"])
        if len(missing_keys) > 0:
            logger.warning(f"Missing values for template string variables {missing_keys}!")

    return new_template


def execute_requests_against_api(
        requests: list[dict],
        api_name: str,
        responses_dir: pathlib.Path,
        history_semaphore, history
) -> list[dict]:
    """Execute a list of requests against one of the APIs.

    Args:
        requests: A list of API requests.
        api_name: The name of the API.

    Returns:
        A list of API responses.
    """
    request_names = []
    for i, request in enumerate(requests):
        request_names.append(f"request_{i}.json")
        request["seed"] = _openai_request_seed

    if api_name == "openai":
        responses = openai_execute(requests, history_semaphore=history_semaphore, history=history, force=0.2, silent=True)
    #elif api_name == "vllm":
    #    return vllm_execute(requests)
    else:
        raise AssertionError(f"Unknown API name '{api_name}'!")
    

    num_failed = 0
    finish_reasons = collections.Counter()
    for response in responses:
        if "choices" in response.keys():
            finish_reasons[response["choices"][0]["finish_reason"]] += 1
        else:
            num_failed += 1

    for key in finish_reasons.keys():
        if key != "stop":
            logger.warning(f"{finish_reasons['length']} generations were stopped due to {key}!")

    if num_failed > 0:
        logger.warning(f"{num_failed} requests failed!")

    return responses


def extract_text_from_response(response: dict) -> str | None:
    """Extract the text from an OpenAI API response.

    Args:
        response: An OpenAI API response.

    Returns:
        The generated text or None if the API request failed.
    """
    if "choices" not in response.keys():
        return None

    return response["choices"][0]["message"]["content"]



def load_json(path: pathlib.Path) -> dict | list:
    """Load the JSON object from the given file path.

    Args:
        path: The pathlib.Path to the JSON file.

    Returns:
        The JSON object.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(obj: dict | list, path: pathlib.Path) -> None:
    """Dump the given JSON object to the given file path.

    Args:
        obj: The JSON object.
        path: The pathlib.Path to the JSON file.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file)