import json
import logging
import math
import multiprocessing
import random
import shutil
import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from wikidbs import utils
from wikidbs.database import Database
from wikidbs.database_creation import create_database_for_topic
from wikidbs.mongodb import get_db_connection
from wikidbs.serialization import converter

log = logging.getLogger(__name__)

#######################################################################################
#### Crawls databases from the WikiDB dump  (without renaming and post-processing)
#### - performance optimized with many processes: please adapt databases.yaml -
#######################################################################################

@hydra.main(version_base=None, config_path="../conf", config_name="databases.yaml")
def start_db_creation(cfg: DictConfig):
    # create data and profiling folder if necessary
    creation_run_path = Path(f"./data/databases/{cfg.creation_run_name}")
    creation_run_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path("./conf/databases.yaml"), creation_run_path)

    # make sure that the profiling jsonl dict exists
    profiling_dict = Path(cfg.wikidata_jsonl)
    assert profiling_dict.exists(), f"Could not find the profiling jsonlines dict file at: {profiling_dict}"

    log.info(f"Starting to create databases, limit is {cfg.limit}, min num rows is {cfg.min_rows}")

    # create property lookup dataframe (to map ids to natural language labels), should be done only once
    db = get_db_connection(cfg)

    before = time.time()
    with open(cfg.wikidata_jsonl, "r") as profiling_file:
        all_lines = profiling_file.readlines()
    log.info(f"Read profiling file in {time.time() - before:.2f} seconds.")  # 25.55s for profiling_info_final.jsonl

    all_lines = all_lines[:cfg.limit]
    random.shuffle(all_lines)
    log.info(f"Planning to crawl {len(all_lines)} databases")

    before = time.time()
    all_statistics = []
    if cfg.processes is None:
        db = get_db_connection(cfg)
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        with open(cfg.wikidata_labels, "r", encoding="utf8") as file:
            wikidata_labels = json.load(file)

        with open(cfg.wikidata_properties, "r", encoding="utf8") as file:
            wikidata_properties = json.load(file)

        p_lookup_df = pd.read_csv(cfg.p_lookup_df)

        for line in tqdm(all_lines, desc="handle lines"):
            stats = handle_line(line, p_lookup_df, embedding_model, db, wikidata_labels, wikidata_properties, creation_run_path, cfg)
            all_statistics.append(stats)
    else:
        #line_group_size = math.ceil(len(topic_dicts_to_process) / cfg.processes)
        line_groups = [[] for _ in range(cfg.processes)]
        for idx, line in enumerate(all_lines):
            line_groups[idx % len(line_groups)].append(line)
        all_params = [(line_group, creation_run_path, cfg) for line_group in line_groups]
        with multiprocessing.Pool(cfg.processes) as pool:
            for stats in pool.imap(wrapper, all_params):
                all_statistics += stats
    log.info(f"Processed {len(all_lines)} lines in {time.time() - before:.2f} seconds.")

    before = time.time()
    statistics = {"number_of_tables": [], "number_of_cols": [], "number_of_rows": [], "table_names_orig": [],
                  "table_names_llm": [], "column_names_orig": [], "column_names_llm": []}
    num_created_databases = 0
    for stats in all_statistics:
        if stats is not None:
            num_created_databases += 1
            if type(stats) == str:
                print("Stats is: ", stats)
            for key, value in stats.items():
                statistics[key] += value
    with open(creation_run_path / "db_statistics.json", "w", encoding="utf-8") as statistics_file:
        json.dump(statistics, statistics_file, indent=2)
    log.info(f"Gathered statistics in {time.time() - before:.2f} seconds.")

    log.info(f"Created {num_created_databases} DBs in {len(all_statistics)} tries.")


def handle_line(
        line: str,
        p_lookup_df: pd.DataFrame,
        embedding_model: SentenceTransformer,
        db,
        wikidata_labels: dict,
        wikidata_properties: dict,
        creation_run_path: Path,
        cfg: DictConfig
) -> dict | None:
    statistics = {"number_of_tables": [], "number_of_cols": [], "number_of_rows": [], "table_names_orig": [],
                  "table_names_llm": [], "column_names_orig": [], "column_names_llm": []}

    topic_dict = json.loads(line)

    # do not process dbs that already have been created
    table_name = topic_dict["predicate_label"] + " " + topic_dict["object_label"]
    table_name = table_name.replace("/", " ")
    db_name = utils.slugify(table_name)
    db_path = Path(creation_run_path /f"{topic_dict['idx']} {db_name}")
    if db_path.exists():
        return None

    num_rows_for_table = topic_dict["num_rows"]

    if cfg.min_rows and num_rows_for_table < cfg.min_rows:
        return None
    if cfg.max_rows and num_rows_for_table > cfg.max_rows:
        return None

    log.debug("################################################################################################")
    log.debug("########################################################################################################")
    log.info(
        f"Database: *{topic_dict['idx']} {topic_dict['predicate_label']} - {topic_dict['object_label']}* with {num_rows_for_table} rows")

    # create_db_for_topic
    database = create_database_for_topic(cfg=cfg,
                                         db=db,
                                         topic_dict=topic_dict,
                                         properties_lookup_df=p_lookup_df,
                                         embedding_model=embedding_model,
                                         wikidata_labels=wikidata_labels,
                                         wikidata_properties=wikidata_properties)

    if database is None:
        return None

    statistics["number_of_tables"].append(len(database.tables))
    statistics["number_of_cols"].append([len(table.table_df.columns) for table in database.tables])
    statistics["number_of_rows"].append([len(table.table_df) for table in database.tables])
    statistics["table_names_orig"].append([table.table_name for table in database.tables])
    statistics["column_names_orig"].append([list(str(x[0]) for x in table.columns) for table in database.tables])

    # serialize database class and tables to disk
    with open(db_path / "database.json", "w", encoding="utf-8") as file:
        for table in database.tables:
            table.rows = None  # cannot serialize rows since dict keys are tuples...
            table.full_properties_with_outgoing_items = None  # cannot serialize...
            table.properties_with_outgoing_items = None  # cannot serialize...
        json.dump(converter.unstructure(database), file)

    with open(db_path / "database.json", "r", encoding="utf-8") as file:
        database_copy = converter.structure(json.load(file), Database)

    return statistics


def handle_line_group(
        line_group: list[str],
        creation_run_path: Path,
        cfg: DictConfig
):
    p_lookup_df = pd.read_csv(cfg.p_lookup_df)

    with open(cfg.wikidata_labels, "r", encoding="utf8") as file:
        wikidata_labels = json.load(file)

    with open(cfg.wikidata_properties, "r", encoding="utf8") as file:
        wikidata_properties = json.load(file)

    log.info("Loaded info files")

    db = get_db_connection(cfg)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return [handle_line(line, p_lookup_df, embedding_model, db, wikidata_labels, wikidata_properties, creation_run_path, cfg) for line in line_group]


def wrapper(args):
    return handle_line_group(*args)


if __name__ == "__main__":
    start_db_creation()