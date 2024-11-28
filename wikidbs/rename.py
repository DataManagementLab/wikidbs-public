import collections
import logging
import re
import unicodedata
from copy import deepcopy
from typing import Literal, Union

import numpy as np
import pandas as pd
import pydantic
from omegaconf import DictConfig, OmegaConf

from wikidbs.database import Database
from wikidbs.table import Table, ForeignKey

log = logging.getLogger(__name__)

_sample_rows_random = np.random.default_rng(seed=351338462)
_openai_request_seed: int = 321164097
postprocess_name_modes = [
    "no_pascal",  # 'CountryName'
    "no_pascal",  # 'CountryName'
    "no_pascal",  # 'CountryName'
    "underscores_lowercase",  # 'country_name'
    "underscores_uppercase",  # 'COUNTRY_NAME'
    "underscores_pascal",  # 'Country_Name'
]


def sample_rows(
        table: pd.DataFrame,
        *other_tables: pd.DataFrame,
        num_rows: int,
        mode: Literal["random"] | Literal["full"]
) -> Union[pd.DataFrame, tuple[pd.DataFrame, ...]]:
    """Sample rows from a pd.DataFrame.

    Does NOT raise if there are not enough rows.

    Args:
        table: The table to sample from.
        num_rows: The number of rows to sample.
        mode: Whether to sample *random* rows or prefer *full* rows.

    Returns:
        A pd.DataFrame with the sampled rows.
    """
    num_rows = min(num_rows, len(table.index))
    if mode == "random":
        if not other_tables:
            return table.sample(n=num_rows, axis=0, random_state=_sample_rows_random, ignore_index=True)
        else:
            ids = _sample_rows_random.choice(len(table.index), num_rows, replace=False)
            l = [df.iloc[ids] for df in (table,) + other_tables]
            for df in l:
                df.reset_index(drop=True, inplace=True)
            return tuple(l)
    elif mode == "full":
        ids = list(range(len(table.index)))
        sparsities = [row.isna().sum() / len(row.index) for _, row in table.iterrows()]
        _sample_rows_random.shuffle(ids)
        ids.sort(key=lambda ix: sparsities[ix])
        ids = ids[:num_rows]
        if not other_tables:
            df = table.iloc[ids]
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            l = [df.iloc[ids] for df in (table,) + other_tables]
            for df in l:
                df.reset_index(drop=True, inplace=True)
            return tuple(l)
    else:
        raise AssertionError(f"Invalid sample_rows mode '{mode}'!")


def fill_template(template: str, **args) -> str:
    """Replace {{variables}} in the template with the given values.

    Args:
        template: The given template string, which may contain {{variables}}.
        **args: The values for the variables.

    Raises in case of missing values, but not in case of unneeded values.

    Returns:
        The template string with {{variables}} replaced by values.
    """

    def replace_variable(match) -> str:
        variable = match.group(1)
        if variable not in args.keys():
            raise AssertionError(f"Missing value for template string variable {variable}!")
        return args[variable]

    return re.sub(r"\{\{([^{}]+)\}\}", replace_variable, template)


def fill_chat_template(
        template: list[dict[str, str] | str],
        **args
) -> list[dict[str, str]]:
    """Replace {{variables}} in the chat template with the given values.

    A value can be a list of messages, a message, or a string, and replacement happens in that order.

    Raises in case of missing values, but not in case of unneeded values.

    Args:
        template: List of template messages containing {{variables}}.
        **args: The given string, message, or list of messages as values for the variables.

    Returns:
        The filled-out template.
    """
    template = deepcopy(template)

    # replace message variables with lists of messages
    for key, value in args.items():
        template_key = "{{" + key + "}}"
        if isinstance(value, list):
            new_template = []
            for message in template:
                if message == template_key:
                    new_template += value
                else:
                    new_template.append(message)
            template = new_template

    # replace message variables with messages
    for key, value in args.items():
        template_key = "{{" + key + "}}"
        if isinstance(value, dict):
            new_template = []
            for message in template:
                if message == template_key:
                    new_template.append(value)
                else:
                    new_template.append(message)
            template = new_template

    # check for missing message values
    for message in template:
        if isinstance(message, str):
            raise AssertionError(f"Missing values for template message variable {message}!")

    # replace string variables with strings, which already checks for missing string values
    str_args = {k: v for k, v in args.items() if isinstance(v, str)}
    for message in template:
        message["content"] = fill_template(message["content"], **str_args)

    return template


def extract_text_from_response(response: dict) -> str | None:
    if "choices" not in response.keys():
        return None

    return response["choices"][0]["message"]["content"]


def determine_cols_to_change_name_of(table):
    return [x for x in table.columns if not "ID" in x[0]]


def prepare_prompt_table(cfg: DictConfig, table: Table):
    # prepare table for request, only include columns that don't contain IDs of something
    cols_to_change_name_of = determine_cols_to_change_name_of(table)
    start_table_df_for_request = sample_rows(
        table.table_df[cols_to_change_name_of],
        num_rows=cfg.num_table_rows_in_prompt,
        mode="full"
    )
    start_table_df_for_request = start_table_df_for_request.map(lambda x: x[0] if isinstance(x, tuple) else x)
    start_table_df_for_request = start_table_df_for_request.map(
        lambda x: str(x[0] if isinstance(x, list) else x)[:cfg.trim_cell_values])
    start_table_df_for_request = start_table_df_for_request.rename(
        columns={x: x[0] for x in start_table_df_for_request.columns}
    )
    return start_table_df_for_request.to_csv(index=False)


def apply_new_column_names(
        new_col_names_list: list[str],
        table: Table,
        cols_to_change_name_of: list
) -> collections.Counter:
    failures = collections.Counter()

    renamed_columns = {}
    if len(cols_to_change_name_of) == len(new_col_names_list):
        for orig_col in table.table_df.columns:
            if orig_col in cols_to_change_name_of:
                col_idx = cols_to_change_name_of.index(orig_col)
                llm_col_name = str(new_col_names_list[col_idx])
                if llm_col_name != "":
                    renamed_columns[orig_col] = (llm_col_name, orig_col[1], orig_col[2])
                else:
                    renamed_columns[orig_col] = orig_col
                    failures["llm_column_name_is_original_column_name_due_to_empty_string"] += 1
            else:
                renamed_columns[orig_col] = orig_col
    else:
        for orig_col in table.table_df.columns:
            renamed_columns[orig_col] = orig_col
            failures["llm_column_name_is_original_column_name_due_to_wrong_length"] += 1

    renamed_table_df = table.table_df.copy()
    renamed_table_df = renamed_table_df.rename(columns=renamed_columns)
    table.llm_renamed_df = renamed_table_df
    return failures


class RenameStartTableResponse(pydantic.BaseModel):
    improved_database_name: str
    improved_table_name: str
    improved_column_names: list[str]


def create_rename_start_table_request(cfg: DictConfig, database: Database) -> dict:
    request = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "seed": _openai_request_seed,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": "wikidbs_rename_start_table_response",
                "schema": {
                    "properties": {
                        "improved_database_name": {
                            "title": "An appropriate name for the database.",
                            "type": "string"
                        },
                        "improved_table_name": {
                            "title": "An appropriate and realistic name for the given table.",
                            "type": "string"
                        },
                        "improved_column_names": {
                            "items": {
                                "type": "string"
                            },
                            "title": "Realistic attribute names for the given table that are related to the database topic domain.",
                            "type": "array"
                        }
                    },
                    "required": ["improved_database_name", "improved_table_name", "improved_column_names"],
                    "additionalProperties": False,
                    "type": "object"
                }
            }
        }
    }

    start_table_linearized = prepare_prompt_table(cfg=cfg, table=database.start_table)

    request["messages"] = fill_chat_template(
        OmegaConf.to_container(cfg.prompt_template_start),
        database_start_table_name=database.start_table.table_name,
        start_table=start_table_linearized
    )

    return request


def rename_start_table(database: Database, response: dict) -> collections.Counter:
    failures = collections.Counter()
    response_text = extract_text_from_response(response)
    if response_text is None:
        failures["request_failed"] += 1
        log.error(f"Request failed: {response}")
        raise RuntimeError(f"Request failed: {response}")

    try:
        parsed_response = RenameStartTableResponse.model_validate_json(response_text)
    except Exception as e:
        failures["parsing_failed"] += 1
        log.error(f"Parsing failed: {response_text}")
        raise e

    # new name for the database
    llm_database_name = parsed_response.improved_database_name.strip()
    log.debug(f"For DB *{database.start_table.table_name}*")
    log.debug(f"LLM database name: *{llm_database_name}*")
    database.llm_db_name = llm_database_name

    llm_start_table_name = parsed_response.improved_table_name.strip()
    log.debug(f"LLM start table name: *{llm_start_table_name}*")
    database.start_table.llm_table_name = llm_start_table_name

    cols_to_change_name_of = determine_cols_to_change_name_of(database.start_table)
    col_names = parsed_response.improved_column_names
    failures += apply_new_column_names(new_col_names_list=col_names,
                                       table=database.start_table,
                                       cols_to_change_name_of=cols_to_change_name_of)
    return failures


class RenameConnectedTableResponse(pydantic.BaseModel):
    improved_table_name: str
    improved_column_names: list[str]


def create_rename_connected_table_request(cfg: DictConfig, database: Database, table_to_rename: Table,
                                          foreign_keys: list[ForeignKey]) -> dict:
    request = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "seed": _openai_request_seed,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "wikidbs_rename_connected_table_response",
                "strict": True,
                "schema": {
                    "properties": {
                        "improved_table_name": {
                            "title": "An appropriate and realistic name for the given table.",
                            "type": "string"
                        },
                        "improved_column_names": {
                            "items": {
                                "type": "string"
                            },
                            "title": "Realistic attribute names for the given table that are related to the database topic domain.",
                            "type": "array"
                        }
                    },
                    "required": ["improved_table_name", "improved_column_names"],
                    "additionalProperties": False,
                    "type": "object"
                }
            }
        }
    }

    fk_table_linearized = prepare_prompt_table(cfg=cfg, table=table_to_rename)

    # prepare foreign key relationships to the given table (strings parent_table.column_name)
    table_foreign_keys = []
    table_foreign_key_columns = []
    for fk in foreign_keys:
        if fk.reference_table_name == table_to_rename.table_name:
            table_foreign_keys.append(f"{fk.source_table_name}.{fk.column_name}")
            table_foreign_key_columns.append(str(fk.column_name))

    request["messages"] = fill_chat_template(
        OmegaConf.to_container(cfg.prompt_template_fks),
        database_start_table_name=database.start_table.table_name,
        database_name=database.llm_db_name,
        fk_table=fk_table_linearized,
        fk_table_name=table_to_rename.table_name,
        fk_relationships=str(table_foreign_keys),
        fk_columns=str(table_foreign_key_columns)
    )

    return request


def rename_connected_table(table_to_rename: Table, response: dict) -> collections.Counter:
    failures = collections.Counter()
    response_text = extract_text_from_response(response)

    try:
        parsed_response = RenameConnectedTableResponse.model_validate_json(response_text)
    except pydantic.ValidationError:
        log.error(f"Parsing failed: {response_text}")
        failures["parsing_failed"] += 1
        failures["llm_table_name_is_original_table_name_due_to_parsing_failed"] += 1
        improved_column_names = []
        for column in table_to_rename.table_df.columns:
            improved_column_names.append(column[0])
            failures["llm_column_name_is_original_column_name_due_to_parsing_failed"] += 1
        parsed_response = RenameConnectedTableResponse(
            improved_table_name=table_to_rename.table_name,
            improved_column_names=improved_column_names
        )

    # new table name
    llm_table_name = parsed_response.improved_table_name.strip()
    log.debug(f"LLM table name: *{llm_table_name}* instead of *{table_to_rename.table_name}*")
    table_to_rename.llm_table_name = llm_table_name

    cols_to_change_name_of = determine_cols_to_change_name_of(table_to_rename)
    failures += apply_new_column_names(new_col_names_list=parsed_response.improved_column_names,
                                       table=table_to_rename,
                                       cols_to_change_name_of=cols_to_change_name_of)
    return failures


def camel_case_split(s):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', s)
    return [m.group(0) for m in matches]


def postprocess_name(name: str, mode: str) -> str:
    # do not allow unicode, taken from https://github.com/django/django/blob/master/django/utils/text.py
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s-]", "", name)

    # split into parts
    name = re.sub(r"[-\s]+", "-", name).strip("-_")
    parts = name.split("-")
    parts = [p for part in parts for p in part.split("_")]
    parts = [p for part in parts for p in camel_case_split(part)]
    parts = [p for p in parts if len(p) > 0]

    assert len(parts) > 0, "There must be at least one name part!"

    # adapt casing
    if mode.endswith("lowercase") or mode.endswith("pascal"):
        parts = [part.lower() for part in parts]
    elif mode.endswith("uppercase"):
        parts = [part.upper() for part in parts]

    if mode.endswith("pascal"):
        parts = [part[0].upper() + part[1:] for part in parts]

    # join parts
    if mode.startswith("no"):
        return "".join(parts)
    elif mode.startswith("spaces"):
        return " ".join(parts)
    elif mode.startswith("underscores"):
        return "_".join(parts)
    elif mode.startswith("hyphen"):
        return "-".join(parts)
    else:
        raise NotImplementedError(f"Invalid renaming mode '{mode}'!")
