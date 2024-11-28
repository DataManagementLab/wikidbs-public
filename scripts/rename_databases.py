import ast
import collections
import hashlib
import itertools
import json
import logging
import random
import shutil
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pqdm.processes import pqdm

from wikidbs.database import Database
from wikidbs.openai_batch import openai_batches_create, openai_batches_execute
from wikidbs.rename import create_rename_start_table_request, rename_start_table, \
    create_rename_connected_table_request, rename_connected_table, postprocess_name_modes, \
    postprocess_name
from wikidbs.serialization import converter

log = logging.getLogger(__name__)

RENAME_START_TABLES_PREFIX = "wikidbs-rename-start-table"
RENAME_CONNECTED_TABLES_PREFIX = "wikidbs-rename-connected-table"


def _load_database(db_path: Path) -> Database:
    with open(db_path / "database.json", "r", encoding="utf-8") as file:
        db = converter.structure(json.load(file), Database)
    for table in db.tables:
        table.table_df.columns = [ast.literal_eval(col) for col in table.table_df.columns]
        if table.llm_renamed_df is not None:
            table.llm_renamed_df.columns = [ast.literal_eval(col) for col in table.llm_renamed_df.columns]
    return db


def _save_database(database: Database, db_path: Path) -> None:
    for table in database.tables:
        table.rows = None  # cannot serialize rows since dict keys are tuples...
        table.full_properties_with_outgoing_items = None  # cannot serialize...
        table.properties_with_outgoing_items = None  # cannot serialize...
    with open(db_path / "database.json", "w", encoding="utf-8") as file:
        json.dump(converter.unstructure(database), file)


def create_rename_start_table_request_for_database(args: tuple[DictConfig, int, Path]) -> dict:
    cfg, db_idx, db_path = args

    db = _load_database(db_path)
    request = create_rename_start_table_request(cfg, db)

    return {
        "custom_id": f"{db_idx}-{db_path.name}-start-table-request",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": request
    }


@hydra.main(version_base=None, config_path="../conf", config_name="rename.yaml")
def create_rename_start_table_request_batches(cfg: DictConfig) -> None:
    log.info(f"Create {RENAME_START_TABLES_PREFIX} request batches for databases from {cfg.input_folder}.")

    # scan databases to rename
    dbs_to_rename = [x for x in sorted(Path(cfg.input_folder).iterdir()) if x.is_dir()][:cfg.limit]
    args = [(cfg, db_idx, db_path) for db_idx, db_path in enumerate(dbs_to_rename)]

    # handle databases
    wrapped_requests = []
    for wrapped_request in pqdm(
            args,
            create_rename_start_table_request_for_database,
            desc="create rename start table request batches",
            n_jobs=cfg.processes,
            exception_behaviour="immediate"
    ):
        wrapped_requests.append(wrapped_request)

    # create batch files
    openai_batches_create(
        wrapped_requests,
        Path(cfg.openai_folder) / RENAME_START_TABLES_PREFIX,
        RENAME_START_TABLES_PREFIX
    )


def rename_start_table_for_database(args: tuple[DictConfig, Path, Path, dict]) -> collections.Counter:
    cfg, db_path, output_path, response = args

    db = _load_database(db_path)
    db.db_name = db_path.name[db_path.name.index(" ") + 1:]

    failures = rename_start_table(db, response)

    output_path.joinpath(db_path.name).mkdir()
    _save_database(db, output_path / db_path.name)
    return failures


@hydra.main(version_base=None, config_path="../conf", config_name="rename.yaml")
def rename_start_tables(cfg: DictConfig) -> None:
    log.info(f"Rename start tables.")
    while True:
        responses = openai_batches_execute(
            Path(cfg.openai_folder) / RENAME_START_TABLES_PREFIX,
            RENAME_START_TABLES_PREFIX
        )

        if responses is None:
            log.error(f"{RENAME_START_TABLES_PREFIX} not fully executed, try again in 600 seconds.")
            time.sleep(600)
        else:
            log.info(f"{RENAME_START_TABLES_PREFIX} fully executed, continue.")
            break

    responses = {r["custom_id"]: r for r in responses}

    dbs_to_rename = [x for x in sorted(Path(cfg.input_folder).iterdir()) if x.is_dir()][:cfg.limit]
    output_path = Path(cfg.output_folder)
    if output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    args = []
    for db_idx, db_path in enumerate(dbs_to_rename):
        args.append((
            cfg,
            db_path,
            output_path,
            responses[f"{db_idx}-{db_path.name}-start-table-request"]["response"]["body"]
        ))

    # handle databases
    failures = collections.Counter()
    for fails in pqdm(
            args,
            rename_start_table_for_database,
            desc="rename start tables",
            n_jobs=cfg.processes,
            exception_behaviour="immediate"
    ):
        failures += fails
    log.info(f"{failures}")


def create_rename_connected_table_request_for_database(args: tuple[DictConfig, int, Path]) -> list[dict]:
    cfg, db_idx, db_path = args

    db = _load_database(db_path)

    res = []
    for ct_idx, table_to_rename in enumerate(db.tables[1:]):
        request = create_rename_connected_table_request(cfg, db, table_to_rename, db.foreign_keys)
        res.append({
            "custom_id": f"{db_idx}-{db_path.name}-{ct_idx}-{table_to_rename.table_name}-connected-table-request",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request
        })
    return res


@hydra.main(version_base=None, config_path="../conf", config_name="rename.yaml")
def create_rename_connected_table_request_batches(cfg: DictConfig) -> None:
    log.info(f"Create {RENAME_CONNECTED_TABLES_PREFIX} request batches for databases from {cfg.input_folder}.")

    # scan databases to rename (dbs with renamed start tables are already in output folder)
    dbs_to_rename = [x for x in sorted(Path(cfg.output_folder).iterdir()) if x.is_dir()][:cfg.limit]
    args = [(cfg, db_idx, db_path) for db_idx, db_path in enumerate(dbs_to_rename)]

    # handle databases
    wrapped_requests = []
    for wrapped_request_for_db in pqdm(
            args,
            create_rename_connected_table_request_for_database,
            desc="create rename connected table request batches",
            n_jobs=cfg.processes,
            exception_behaviour="immediate"
    ):
        wrapped_requests += wrapped_request_for_db

    # create batch files
    openai_batches_create(
        wrapped_requests,
        Path(cfg.openai_folder) / RENAME_CONNECTED_TABLES_PREFIX,
        RENAME_CONNECTED_TABLES_PREFIX
    )


def rename_connected_tables_for_database(args: tuple[DictConfig, int, Path, dict]) -> collections.Counter:
    cfg, db_idx, db_path, responses = args

    db = _load_database(db_path)
    failures = collections.Counter()
    for ct_idx, table_to_rename in enumerate(db.tables[1:]):
        response = responses[f"{db_idx}-{db_path.name}-{ct_idx}-{table_to_rename.table_name}-connected-table-request"]
        failures += rename_connected_table(table_to_rename, response)

    _save_database(db, db_path)
    return failures


@hydra.main(version_base=None, config_path="../conf", config_name="rename.yaml")
def rename_connected_tables(cfg: DictConfig) -> None:
    log.info(f"Rename connected tables.")
    while True:
        responses = openai_batches_execute(
            Path(cfg.openai_folder) / RENAME_CONNECTED_TABLES_PREFIX,
            RENAME_CONNECTED_TABLES_PREFIX
        )

        if responses is None:
            log.error(f"{RENAME_CONNECTED_TABLES_PREFIX} not fully executed, try again in 600 seconds.")
            time.sleep(600)
        else:
            log.info(f"{RENAME_CONNECTED_TABLES_PREFIX} fully executed, continue.")
            break

    responses = {r["custom_id"]: r for r in responses}

    dbs_to_rename = [x for x in sorted(Path(cfg.output_folder).iterdir()) if x.is_dir()][:cfg.limit]

    args = []
    for db_idx, db_path in enumerate(dbs_to_rename):
        responses_for_db = {}
        for key, value in responses.items():
            if key.startswith(f"{db_idx}-{db_path.name}-") and key.endswith("-connected-table-request"):
                responses_for_db[key] = value["response"]["body"]
        args.append((
            cfg,
            db_idx,
            db_path,
            responses_for_db
        ))

    # handle databases
    failures = collections.Counter()
    for fails in pqdm(
            args,
            rename_connected_tables_for_database,
            desc="rename connected tables",
            n_jobs=cfg.processes,
            exception_behaviour="immediate"
    ):
        failures += fails
    log.info(f"{failures}")


def postprocess_names_for_database(args: tuple[DictConfig, Path]) -> collections.Counter:
    cfg, db_path = args

    db = _load_database(db_path)
    failures = collections.Counter()

    postprocess_names_random = random.Random(hashlib.sha256(bytes(db.llm_db_name, "utf-8")).digest())
    mode = postprocess_names_random.choice(postprocess_name_modes)

    db.llm_only_db_name = db.llm_db_name
    db.llm_db_name = postprocess_name(db.llm_db_name, mode=mode)

    prev_llm_table_names = set()
    for table in db.tables:
        table.llm_only_table_name = table.llm_table_name
        # create unique llm_table_name
        name = postprocess_name(table.llm_table_name, mode=mode)
        if name.lower() not in prev_llm_table_names:
            table.llm_table_name = name
        else:
            name = postprocess_name(table.table_name, mode=mode)
            if name.lower() not in prev_llm_table_names:
                table.llm_table_name = name
                failures["duplicate_llm_table_name_use_postprocessed_original_table_name"] += 1
            else:
                failures["duplicate_llm_table_name_use_counting_index"] += 1
                for idx in itertools.count(start=1):
                    name = postprocess_name(f"{table.llm_table_name}_{idx}", mode=mode)
                    if name.lower() not in prev_llm_table_names:
                        table.llm_table_name = name
                        break
        prev_llm_table_names.add(table.llm_table_name.lower())

        prev_llm_col_names = set()
        new_col_names = []
        table.llm_only_column_names = []
        for col, orig_col in zip(table.llm_renamed_df.columns, table.table_df.columns):
            table.llm_only_column_names.append(col[0])

            name = postprocess_name(col[0], mode=mode)
            if name.lower() not in prev_llm_col_names:
                new_col_names.append((name, col[1], col[2]))
                prev_llm_col_names.add(name.lower())
            else:
                name = postprocess_name(orig_col[0], mode=mode)
                if name.lower() not in prev_llm_col_names:
                    new_col_names.append((name, col[1], col[2]))
                    prev_llm_col_names.add(name.lower())
                    failures["duplicate_llm_column_name_use_postprocessed_original_column_name"] += 1
                else:
                    failures["duplicate_llm_column_name_use_counting_index"] += 1
                    for idx in itertools.count(start=1):
                        name = postprocess_name(f"{col[0]}_{idx}", mode=mode)
                        if name.lower() not in prev_llm_col_names:
                            new_col_names.append((name, col[1], col[2]))
                            prev_llm_col_names.add(name.lower())
                            break

        table.llm_renamed_df.columns = new_col_names

    _save_database(db, db_path)
    return failures


@hydra.main(version_base=None, config_path="../conf", config_name="rename.yaml")
def postprocess_names(cfg: DictConfig) -> None:
    log.info(f"Postprocess names.")

    dbs_to_rename = [x for x in sorted(Path(cfg.output_folder).iterdir()) if x.is_dir()][:cfg.limit]

    args = [(cfg, db_path) for db_path in dbs_to_rename]

    # handle databases
    failures = collections.Counter()
    for fails in pqdm(
            args,
            postprocess_names_for_database,
            desc="postprocess names",
            n_jobs=cfg.processes,
            exception_behaviour="immediate"
    ):
        failures += fails
    log.info(f"{failures}")


if __name__ == "__main__":
    create_rename_start_table_request_batches()
    rename_start_tables()
    create_rename_connected_table_request_batches()
    rename_connected_tables()
    postprocess_names()
