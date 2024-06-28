import json
import logging
import math
import multiprocessing
import multiprocessing.dummy
import time
from pathlib import Path
import ast

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from wikidbs.database import Database
from wikidbs.openai import openai_cost_for_cache
from wikidbs.rename import rename
from wikidbs.serialization import converter
from wikidbs.utils import postprocess_name, postprocess_names_random, postprocess_name_modes, is_camel_case_naive


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="rename.yaml")
def start_renaming(cfg: DictConfig):

    # create data and profiling folder if necessary
    output_path = Path(f"{cfg.output_folder}")
    output_path.mkdir(parents=True, exist_ok=True)

    dbs_to_rename = [x for x in list(Path(cfg.input_folder).iterdir()) if x.is_dir()]
    dbs_to_rename = dbs_to_rename[:cfg.limit]

    log.info(f"Starting to rename databases from {cfg.input_folder}, limit is {cfg.limit}, plan to rename {(len(dbs_to_rename))}")

    total_cost_before = openai_cost_for_cache()

    before = time.time()
    if cfg.processes is None:
        for db_folder in tqdm(dbs_to_rename, "handle dbs"):
            dummy_semaphore = multiprocessing.dummy.Semaphore()
            dummy_history =  {"rpm_budget": None, "tpm_budget": None, "last_update": None}
            handle_db(db_folder=db_folder, 
                      output_path=output_path,
                      history_semaphore=dummy_semaphore,
                      history=dummy_history, 
                      cfg=cfg)
            handle_db(db_folder=db_folder, output_path=output_path, cfg=cfg)
    else:
        with multiprocessing.Manager() as manager:
            history_semaphore = manager.Semaphore()
            history = manager.dict(rpm_budget=None, tpm_budget=None, last_update=None)
            line_group_size = math.ceil(len(dbs_to_rename) / cfg.processes)
            line_groups = []
            for left in range(0, len(dbs_to_rename), line_group_size):
                line_groups.append(dbs_to_rename[left:left + line_group_size])
            all_params = [(line_group, output_path, history_semaphore, history, cfg) for line_group in line_groups]
            with multiprocessing.Pool(cfg.processes) as pool:
                for _ in pool.imap(wrapper, all_params):
                    pass # currently doesn't return anything
    log.info(f"Processed lines in {time.time() - before:.2f} seconds.")

    before = time.time()

    total_cost_after = openai_cost_for_cache()
    log.info(f"Done renaming. This cost ${total_cost_after - total_cost_before:.4f}.")


def handle_db(
        db_folder: Path,
        output_path: Path,
        history_semaphore,
        history,
        cfg: DictConfig
) -> dict | None:
    
    # check if database has already been renamed
    if output_path / db_folder.name in list(output_path.iterdir()):
        log.info(f"DB {db_folder.name} has already be renamed")
        return

    try:

        log.info("################################################################################################")
        log.info(f"Database: *{db_folder.name}")

        # load database
        with open(db_folder / "database.json", "r", encoding="utf-8") as file:
            database_to_rename = converter.structure(json.load(file), Database)

        if len(database_to_rename.tables) > 20:
            log.info(f"DB has more than 20 tables, not renaming for now")
            return

        # convert dataframe tuples back to real tuples:
        for table in database_to_rename.tables:
            table.table_df.columns = [ast.literal_eval(col) for col in table.table_df.columns]

        # rename
        try:
            rename(cfg=cfg, database=database_to_rename, responses_dir=output_path, history_semaphore=history_semaphore, history=history)
        except Exception as e:
            if isinstance(e, AssertionError):
                raise AssertionError(e)
            if isinstance(e, RuntimeError):
                raise RuntimeError(e)
            if isinstance(e, NotImplementedError):
                raise NotImplementedError(e)

            log.error(f"Exception: {e}, couldn't rename database {db_folder.name}")
            return

        # create output folder for db
        db_output_path = Path(output_path / db_folder.name)
        db_tables_path = db_output_path / "tables"
        db_tables_path.mkdir(parents=True)

        # standardize the casing of all table and column names in the database
        mode = postprocess_names_random.choice(postprocess_name_modes)  
        log.debug(f"Chosen case mode: {mode}")

        # replace llm_dataframe of each table with the dataframe containing only the labels
        for table in database_to_rename.tables:
            # keep camel case table names as they are, naive check:
            if not is_camel_case_naive(table.llm_table_name[0]):
                new_table_name = postprocess_name(table.llm_table_name, mode=mode)
            else: 
                new_table_name = table.llm_table_name
            
            table.llm_table_name = new_table_name

            new_col_names = []
            for col in table.llm_renamed_df.columns:
                new_col_name = postprocess_name(col, mode=mode)
                new_col_names.append(new_col_name)
            table.llm_renamed_df.columns = new_col_names

            # save renaming info cased
            renaming_info = {}
            for table in database_to_rename.tables:
                table_info = {}
                try:
                    table_info[table.table_name] = table.llm_table_name
                    column_info = {}
                    for col_idx, column in enumerate(table.table_df.columns):
                        column_info[column[0]] = table.llm_renamed_df.columns[col_idx][0]
                    table_info["columns"] = column_info
                except AttributeError:
                    # not all tables might have been renamed
                    pass
                renaming_info[table.table_name] = table_info
            with open(db_folder / "llm_renaming_cased.json", "w", encoding="utf-8") as renaming_file:
                json.dump(renaming_info, renaming_file, indent=2)

        database_to_rename.tables_to_csv(db_tables_path, use_llm_names=False)

        with open(db_output_path / "database.json", "w", encoding="utf-8") as file:
            for table in database_to_rename.tables:
                table.rows = None  # cannot serialize rows since dict keys are tuples...
                table.full_properties_with_outgoing_items = None  # cannot serialize...
                table.properties_with_outgoing_items = None  # cannot serialize...
            json.dump(converter.unstructure(database_to_rename), file)

        with open(db_output_path / "metadata.json", "w", encoding="utf-8") as metadata_file:
            json.dump({ "db_name": database_to_rename.db_name,
                        "num_tables": len(database_to_rename.tables),
                        "num_cols": [len(table.table_df.columns) for table in database_to_rename.tables],
                        "num_rows": [len(table.table_df) for table in database_to_rename.tables]}, metadata_file, indent=2)
    except:
        log.warning(f"Failed to rename database {db_folder.name}", exc_info=True)

    return 


def handle_line_group(
        line_group: list[str],
        output_path: Path,
        history_semaphore,
        history,
        cfg: DictConfig
):
    return [handle_db(db_folder=db_path, output_path=output_path, history_semaphore=history_semaphore, history=history, cfg=cfg) for db_path in line_group]


def wrapper(args):
    return handle_line_group(*args)


if __name__ == "__main__":
    start_renaming()
