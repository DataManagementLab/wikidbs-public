########################################################################################################################
# OpenAI API batch helpers version: 2024-10-10
########################################################################################################################
import collections
import json
import logging
import os
import pathlib
import shutil

import requests
import tqdm

from wikidbs.openai import _Request, _Response, openai_execute

logger = logging.getLogger(__name__)


########################################################################################################################
# API
########################################################################################################################

def openai_batches_create(
        wrapped_requests: list[dict],
        openai_dir: pathlib.Path,
        unique_prefix: str
) -> None:
    """Create batch files for the given requests.

    This method also computes the maximum cost incurred by the requests.
    The order of the requests in the batches corresponds to the original order of the requests.

    Args:
        wrapped_requests: A list of wrapped API requests.
        openai_dir: A pathlib.Path at which to store the batch requests/responses/...
        unique_prefix: A unique prefix.
    """
    if openai_dir.is_dir():
        logger.info(f"press enter to continue and delete existing batch requests/responses in `{openai_dir}`")
        input(f"press enter to continue and delete existing batch requests/responses in `{openai_dir}`")
        shutil.rmtree(openai_dir)
    batches_dir = openai_dir / "0_request_batches"
    batches_dir.mkdir(parents=True)

    batch_ix, batch_file_size, batch_file_reqs, cost = 0, 0, 0, 0
    with tqdm.tqdm(
            total=len(wrapped_requests),
            desc=f"create batches `{unique_prefix}`",
            postfix=_openai_create_batches_postfix(batch_ix, cost)
    ) as progress_bar:
        batch_file = open(batches_dir / f"{unique_prefix}-batch-{batch_ix}.jsonl", "w", encoding="utf-8")
        for wrapped_request in wrapped_requests:
            request = _Request(wrapped_request["body"])
            request.check()
            cost += request.max_cost() / 2  # batch API costs half as much
            wrapped_req_str = json.dumps(wrapped_request)
            wrapped_req_size = len(wrapped_req_str.encode("utf-8")) + len("\n".encode("utf-8"))
            if batch_file_size + wrapped_req_size > 98_000_000 or batch_file_reqs + 1 > 50_000:
                batch_file.close()
                batch_ix += 1
                batch_file = open(batches_dir / f"{unique_prefix}-batch-{batch_ix}.jsonl", "w", encoding="utf-8")
                batch_file_size, batch_file_reqs = 0, 0
            batch_file.write(wrapped_req_str)
            batch_file.write("\n")
            batch_file_size += wrapped_req_size
            batch_file_reqs += 1
            progress_bar.update()
            progress_bar.set_postfix(_openai_create_batches_postfix(batch_ix, cost))

    batch_file.close()


def openai_batches_execute(
        openai_dir: pathlib.Path,
        unique_prefix: str
) -> list[dict] | None:
    """Start batches at OpenAI.

    Args:
        openai_dir: A pathlib.Path at which to store the requests/responses/...
        unique_prefix: A unique prefix.

    Returns:
        List of wrapped responses upon completion, else None.
    """
    success = True
    batches_dir = openai_dir / "0_request_batches"
    if not batches_dir.is_dir():
        raise FileNotFoundError(f"Missing request batches directory `{batches_dir}`!")
    upload_responses_dir = openai_dir / "1_upload_responses"
    upload_responses_dir.mkdir(parents=True, exist_ok=True)
    start_responses_dir = openai_dir / "2_start_responses"
    start_responses_dir.mkdir(parents=True, exist_ok=True)
    retrieve_responses_dir = openai_dir / "3_retrieve_responses"
    retrieve_responses_dir.mkdir(parents=True, exist_ok=True)
    response_batches_dir = openai_dir / "4_response_batches"
    response_batches_dir.mkdir(parents=True, exist_ok=True)
    error_batches_dir = openai_dir / "5_error_batches"
    error_batches_dir.mkdir(parents=True, exist_ok=True)
    todo_delete_input_dir = openai_dir / "6_todo_delete_input"
    todo_delete_input_dir.mkdir(parents=True, exist_ok=True)
    todo_delete_output_dir = openai_dir / "7_todo_delete_output"
    todo_delete_output_dir.mkdir(parents=True, exist_ok=True)
    todo_delete_error_dir = openai_dir / "8_todo_delete_error"
    todo_delete_error_dir.mkdir(parents=True, exist_ok=True)

    all_wrapped_responses = {}
    all_wrapped_errors = {}
    batch_paths = list(sorted(batches_dir.glob("*.jsonl"), key=lambda p: int(p.name[p.name.rindex("-") + 1:-6])))
    with _BatchesProgressBar(total=len(batch_paths), desc=f"execute batches `{unique_prefix}`") as progress_bar:
        for batch_path in batch_paths:
            ############################################################################################################
            # upload batch
            ############################################################################################################
            upload_response_path = upload_responses_dir / batch_path.name[:-1]
            if not upload_response_path.is_file():
                with open(batch_path, "rb") as file:
                    http_response = requests.post(
                        url="https://api.openai.com/v1/files",
                        files={"file": file},
                        data={"purpose": "batch"},
                        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
                    )

                match http_response.status_code:
                    case 200:
                        progress_bar.uploaded += 1
                        progress_bar.update_postfix()
                        with open(upload_response_path, "w", encoding="utf-8") as file:
                            json.dump(http_response.json(), file)
                    case _:
                        logger.warning(f"failed to upload `{batch_path}`: {http_response}")
                        success = False
            else:
                progress_bar.uploaded += 1
                progress_bar.update_postfix()

            ############################################################################################################
            # start batch at OpenAI
            ############################################################################################################
            start_response_path = start_responses_dir / batch_path.name[:-1]
            if upload_response_path.is_file() and not start_response_path.is_file():
                with open(upload_response_path, "r", encoding="utf-8") as file:
                    upload_response = json.load(file)
                with open(batch_path, "r", encoding="utf-8") as file:
                    first_wrapped_request = json.loads(file.readline())

                http_response = requests.post(
                    url="https://api.openai.com/v1/batches",
                    json={
                        "input_file_id": upload_response["id"],
                        "endpoint": first_wrapped_request["url"],
                        "completion_window": "24h"
                    },
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
                )

                match http_response.status_code:
                    case 200:
                        progress_bar.started += 1
                        progress_bar.update_postfix()
                        with open(start_response_path, "w", encoding="utf-8") as file:
                            json.dump(http_response.json(), file)
                    case _:
                        logger.warning(f"failed to start `{batch_path}`: {http_response}")
                        success = False
            else:
                progress_bar.started += 1
                progress_bar.update_postfix()

            ############################################################################################################
            # retrieve batch status
            ############################################################################################################
            retrieve_response_path = retrieve_responses_dir / batch_path.name[:-1]
            response_batch_todo_path = response_batches_dir / f"{batch_path.name[:-1]}.todo"
            error_batch_todo_path = error_batches_dir / f"{batch_path.name[:-1]}.todo"
            delete_input_todo_path = todo_delete_input_dir / f"{batch_path.name[:-1]}.todo"
            if start_response_path.is_file() and not retrieve_response_path.is_file():
                with open(start_response_path, "r", encoding="utf-8") as file:
                    start_response = json.load(file)

                http_response = requests.get(
                    url=f"https://api.openai.com/v1/batches/{start_response['id']}",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
                )
                json_response = http_response.json()

                progress_bar.status[json_response["status"]] += 1
                progress_bar.update_postfix()

                match http_response.status_code, json_response["status"]:
                    case 200, "completed" | "failed" | "cancelled" | "expired":  # TODO: deal with expired batches
                        if json_response["input_file_id"] is not None:
                            with open(delete_input_todo_path, "w", encoding="utf-8") as file:
                                json.dump({"file_id": json_response["input_file_id"]}, file)
                        if json_response["output_file_id"] is not None:
                            with open(response_batch_todo_path, "w", encoding="utf-8") as file:
                                json.dump({"file_id": json_response["output_file_id"]}, file)
                        if json_response["error_file_id"] is not None:
                            with open(error_batch_todo_path, "w", encoding="utf-8") as file:
                                json.dump({"file_id": json_response["error_file_id"]}, file)
                        with open(retrieve_response_path, "w", encoding="utf-8") as file:
                            json.dump(json_response, file)
                    case 200, _:
                        success = False
                    case _:
                        logger.warning(f"failed to retrieve `{batch_path}`: {http_response}")
                        success = False
            elif retrieve_response_path.is_file():
                with open(retrieve_response_path, "r", encoding="utf-8") as file:
                    retrieve_response = json.load(file)
                progress_bar.status[retrieve_response["status"]] += 1
                progress_bar.update_postfix()

            ############################################################################################################
            # delete input file at OpenAI
            ############################################################################################################
            if delete_input_todo_path.is_file():
                with open(delete_input_todo_path, "r", encoding="utf-8") as file:
                    delete_input_todo = json.load(file)

                http_response = requests.delete(
                    url=f"https://api.openai.com/v1/files/{delete_input_todo['file_id']}",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
                )

                match http_response.status_code:
                    case 200:
                        os.remove(delete_input_todo_path)
                    case _:
                        logger.warning(f"failed to delete input for `{batch_path}`: {http_response}")
                        success = False

            ############################################################################################################
            # download responses
            ############################################################################################################
            response_batch_path = response_batches_dir / batch_path.name
            delete_output_todo_path = todo_delete_output_dir / f"{batch_path.name[:-1]}.todo"
            if response_batch_todo_path.is_file():
                with open(response_batch_todo_path, "r", encoding="utf-8") as file:
                    response_batch_todo = json.load(file)

                http_response = requests.get(
                    url=f"https://api.openai.com/v1/files/{response_batch_todo['file_id']}/content",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
                )

                match http_response.status_code:
                    case 200:
                        with open(response_batch_path, "wb") as file:
                            file.write(http_response.content)
                        with open(delete_output_todo_path, "w", encoding="utf-8") as file:
                            json.dump(response_batch_todo, file)
                        os.remove(response_batch_todo_path)
                    case _:
                        logger.warning(f"failed to download response batch `{batch_path}`: {http_response}")
                        success = False

            ############################################################################################################
            # download errors
            ############################################################################################################
            error_batch_path = error_batches_dir / batch_path.name
            delete_error_todo_path = todo_delete_error_dir / f"{batch_path.name[:-1]}.todo"
            if error_batch_todo_path.is_file():
                with open(error_batch_todo_path, "r", encoding="utf-8") as file:
                    error_batch_todo = json.load(file)

                http_response = requests.get(
                    url=f"https://api.openai.com/v1/files/{error_batch_todo['file_id']}/content",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
                )

                match http_response.status_code:
                    case 200:
                        with open(error_batch_path, "wb") as file:
                            file.write(http_response.content)
                        with open(delete_error_todo_path, "w", encoding="utf-8") as file:
                            json.dump(error_batch_todo, file)
                        os.remove(error_batch_todo_path)
                    case _:
                        logger.warning(f"failed to download error batch `{batch_path}`: {http_response}")
                        success = False

            ############################################################################################################
            # delete output file at OpenAI
            ############################################################################################################
            if delete_output_todo_path.is_file():
                with open(delete_output_todo_path, "r", encoding="utf-8") as file:
                    delete_output_todo = json.load(file)

                http_response = requests.delete(
                    url=f"https://api.openai.com/v1/files/{delete_output_todo['file_id']}",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
                )

                match http_response.status_code:
                    case 200:
                        os.remove(delete_output_todo_path)
                    case _:
                        logger.warning(f"failed to delete output for `{batch_path}`: {http_response}")
                        success = False

            ############################################################################################################
            # delete error file at OpenAI
            ############################################################################################################
            if delete_error_todo_path.is_file():
                with open(delete_error_todo_path, "r", encoding="utf-8") as file:
                    delete_error_todo = json.load(file)

                http_response = requests.delete(
                    url=f"https://api.openai.com/v1/files/{delete_error_todo['file_id']}",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
                )

                match http_response.status_code:
                    case 200:
                        os.remove(delete_error_todo_path)
                    case _:
                        logger.warning(f"failed to delete error for `{batch_path}`: {http_response}")
                        success = False

            ############################################################################################################
            # gather responses and errors
            ############################################################################################################
            if response_batch_path.is_file():
                with open(response_batch_path, "r", encoding="utf-8") as file:
                    for line in file:
                        wrapped_response = json.loads(line)
                        response = _Response(wrapped_response["response"]["body"])
                        progress_bar.cost += response.total_cost() / 2  # batch API costs half as much
                        if not response.was_successful:
                            logger.warning(f"request failed: {wrapped_response}")
                            progress_bar.req_failed += 1
                        all_wrapped_responses[wrapped_response["custom_id"]] = wrapped_response

            if error_batch_path.is_file():
                with open(error_batch_path, "r", encoding="utf-8") as file:
                    for line in file:
                        wrapped_error = json.loads(line)
                        logger.warning(f"request failed with error: {wrapped_error}")
                        progress_bar.req_failed += 1
                        all_wrapped_errors[wrapped_error["custom_id"]] = wrapped_error

            progress_bar.update()

        if not success:
            return None

        wrapped_responses = []
        for batch_path in batch_paths:
            with open(batch_path, "r", encoding="utf-8") as file:
                for line in file:
                    wrapped_request = json.loads(line)
                    if wrapped_request["custom_id"] in all_wrapped_responses.keys():
                        wrapped_response = all_wrapped_responses[wrapped_request["custom_id"]]
                    elif wrapped_request["custom_id"] in all_wrapped_errors.keys():
                        wrapped_response = all_wrapped_errors[wrapped_request["custom_id"]]
                    else:
                        logger.warning(
                            f"no response/error for `{wrapped_request['custom_id']}`, return empty wrapped response"
                        )
                        wrapped_response = {}
                    wrapped_responses.append(wrapped_response)
        return wrapped_responses


def openai_batches_execute_dummy(
        openai_dir: pathlib.Path,
        unique_prefix: str,
        *,
        force: float | None = None,
        silent: bool = False,
        global_context: dict | None = None,
        global_semaphore: "multiprocessing.Semaphore | None" = None
) -> list[dict] | None:
    """Execute requests from batches using the normal API. Does not write responses/errors... in openai_dir.

    Args:
        openai_dir: A pathlib.Path at which to store the requests/responses/...
        unique_prefix: A unique prefix.
        force: An optional float specifying the cost below or equal to which no confirmation should be required.
        silent: Whether to display log messages and progress bars.
        global_context: Optional global context for use with multiprocessing.
        global_semaphore: Optional global semaphore for use with multiprocessing.

    Returns:
        List of wrapped responses upon completion, else None.
    """
    batches_dir = openai_dir / "0_request_batches"
    if not batches_dir.is_dir():
        raise FileNotFoundError(f"Missing request batches directory `{batches_dir}`!")

    batch_paths = list(sorted(batches_dir.glob("*.jsonl"), key=lambda p: int(p.name[p.name.rindex("-") + 1:-6])))
    requests, custom_ids = [], []
    for batch_path in batch_paths:
        with open(batch_path, "r", encoding="utf-8") as file:
            for line in file:
                wrapped_request = json.loads(line)
                requests.append(wrapped_request["body"])
                custom_ids.append(wrapped_request["custom_id"])

    responses = openai_execute(
        requests,
        force=force,
        silent=silent,
        global_context=global_context,
        global_semaphore=global_semaphore
    )
    return [{"custom_id": cid, "response": {"body": response}} for cid, response in zip(custom_ids, responses)]


########################################################################################################################
# implementation
########################################################################################################################


def _openai_create_batches_postfix(batch_ix: int, cost: float) -> dict[str, str]:
    return {
        "batches": f"{batch_ix + 1}",
        "cost": f"${cost:.2f}"
    }


class _BatchesProgressBar(tqdm.tqdm):
    uploaded: int
    started: int
    status: collections.Counter
    req_failed: int
    cost: float

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.uploaded = 0
        self.started = 0
        self.status = collections.Counter()
        self.req_failed = 0
        self.cost = 0
        self.update_postfix()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return super().__exit__(exc_type, exc_value, traceback)

    def update(self, *args, **kwargs) -> None:
        self.update_postfix()
        super().update(*args, **kwargs)

    def update_postfix(self) -> None:
        parts = [f"uploaded={self.uploaded}", f"started={self.started}"]
        for k, v in self.status.items():
            parts.append(f"{k}={v}")
        parts.append(f"req_failed={self.req_failed}")
        parts.append(f"cost=${self.cost:.2f}")
        self.set_postfix_str(", ".join(parts))
