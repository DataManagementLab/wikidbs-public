import hydra
from omegaconf import DictConfig

import shutil
import logging
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import multiprocessing.dummy

from sentence_transformers import SentenceTransformer

import wikidbs.schema
from wikidbs import utils
from wikidbs.database import Database
from wikidbs.database_creation import create_database_for_topic
from wikidbs.mongodb import get_db_connection
from wikidbs.visualize import create_schema_diagram
from wikidbs.rename import rename
from wikidbs.serialization import converter
from wikidbs.utils import postprocess_name, postprocess_names_random, postprocess_name_modes, is_camel_case_naive, majority_type

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="databases.yaml")
def start_db_creation(cfg: DictConfig):
    # create data and profiling folder if necessary
    creation_run_path = Path(f"./data/databases/{cfg.creation_run_name}")
    creation_run_path.mkdir(parents=True)
    shutil.copy(Path("./conf/databases.yaml"), creation_run_path)

    # get database connection
    db = get_db_connection(cfg)

    with open(cfg.wikidata_labels, "r", encoding="utf8") as file:
        wikidata_labels = json.load(file)

    with open(cfg.wikidata_properties, "r", encoding="utf8") as file:
        wikidata_properties = json.load(file)

    p_lookup_df = pd.read_csv(cfg.p_lookup_df)

    
    # make sure that the profiling jsonl dict exists
    profiling_dict = Path(cfg.wikidata_jsonl)
    assert profiling_dict.exists(), f"Could not find the profiling jsonlines dict file at: {profiling_dict}"

    log.info(f"Starting to create databases, limit is {cfg.limit}, min num rows is {cfg.min_rows}")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    num_created_databases = 0

    statistics = {"number_of_tables": [], "number_of_cols": [], "number_of_rows": [], "table_names_orig": [], "table_names_llm": []}

    # load main table topics from profiling jsonlines script line by line
    with open(cfg.wikidata_jsonl, "r") as profiling_file:
        for line_idx, line in enumerate(tqdm(profiling_file)):
            line = profiling_file.readline()
            topic_dict = json.loads(line)
            num_rows_for_table = topic_dict["num_rows"]

            if cfg.min_rows and num_rows_for_table < cfg.min_rows :
                continue
            if cfg.max_rows and num_rows_for_table > cfg.max_rows:
                continue

            log.info("################################################################################################")
            log.info("########################################################################################################")
            log.info(f"Database: *{topic_dict['predicate_label']} - {topic_dict['object_label']}* at index {line_idx} with {num_rows_for_table} rows")

            # create_db_for_topic
            database = create_database_for_topic(cfg=cfg,
                                                db=db,
                                                topic_dict=topic_dict,
                                                properties_lookup_df=p_lookup_df,
                                                embedding_model=embedding_model,
                                                wikidata_labels=wikidata_labels,
                                                wikidata_properties=wikidata_properties)

            if database is not None:
                num_created_databases +=1

                # save all tables as csv files
                db_name = utils.slugify(database.start_table.table_name)
                db_path = Path(creation_run_path / db_name)
                db_path.mkdir(parents=True)

                # rename
                dummy_semaphore = multiprocessing.dummy.Semaphore()
                dummy_history =  {"rpm_budget": None, "tpm_budget": None, "last_update": None}
                rename(cfg=cfg, database=database, responses_dir=db_path, history_semaphore=dummy_semaphore, history=dummy_history)

                # keep original table names if LLM created duplicates
                llm_table_names = []
                for table in database.tables:
                    llm_table_names.append(table.llm_table_name.lower())
                
                if len(set(llm_table_names)) != len(llm_table_names):
                    for table in database.tables:
                        table.llm_table_name = table.table_name

                # standardize the casing of all table and column names in the database
                mode = postprocess_names_random.choice(postprocess_name_modes)  
                log.debug(f"Chosen case mode: {mode}")

                # replace llm_dataframe of each table with the dataframe containing only the labels
                for table in database.tables:
                    # keep camel case table names as they are, naive check:
                    if not is_camel_case_naive(table.llm_table_name[0]):
                        new_table_name = postprocess_name(table.llm_table_name, mode=mode)
                    else: 
                        new_table_name = table.llm_table_name
                    
                    table.llm_table_name = new_table_name

                    new_col_names = []
                    for col in table.llm_renamed_df.columns:
                        # col is a triple (col_name, property_id, 'datatype')
                        new_col_name = postprocess_name(col[0], mode=mode)
                        new_col_names.append((new_col_name, col[1], col[2]))
                    table.llm_renamed_df.columns = new_col_names

                    # save renaming info cased
                    renaming_info = {}
                    for table in database.tables:
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
                    with open(db_path / "llm_renaming_cased.json", "w", encoding="utf-8") as renaming_file:
                        json.dump(renaming_info, renaming_file, indent=2)

                # create schema information
                final_tables = []
                col_names_first_col = {}
                columns_per_table = {}

                for table_idx, table in enumerate(database.tables):
                    orig_table = database.tables[table_idx]

                    # Determine the datatype for each column
                    column_types = {col: majority_type(orig_table.table_df[col]) for col in orig_table.table_df.columns}
                    table_columns = []

                    for col_idx, col_name in enumerate(table.llm_renamed_df.columns):
                        col_name = col_name[0]
                        orig_column = orig_table.table_df.columns[col_idx]
                        col_datatype = column_types[orig_column]
                        col_pid = orig_table.table_df.columns[col_idx][1]
                        if col_idx <= 1:
                            col_pid = None
                        column = wikidbs.schema.Column(column_name = col_name,
                                                    wikidata_property_id=col_pid,
                                                    data_type = col_datatype)
                        table_columns.append(column)
                                    
                        if col_idx == 0:
                            col_names_first_col[table.llm_table_name] = col_name
                    columns_per_table[table.llm_table_name] = table_columns

                Path(db_path / "tables").mkdir()
                Path(db_path / "tables_with_item_ids").mkdir()

                for table_idx, table in enumerate(database.tables):
                    # handle foreign keys (need to update fk names after paraphrasing)
                    foreign_keys = []
                    for fk in table.foreign_keys:
                        orig_fk_col = fk.column_name
                        fk_col = renaming_info[table.table_name]["columns"][orig_fk_col]

                        orig_fk_reference_table = fk.reference_table_name
                        fk_reference_table = renaming_info[orig_fk_reference_table][orig_fk_reference_table]

                        reference_col = col_names_first_col[fk_reference_table]

                        schema_fk = wikidbs.schema.ForeignKey(column=fk_col,
                                                            reference_column=reference_col,
                                                            reference_table=fk_reference_table)
                        foreign_keys.append(schema_fk)
                    schema_table = wikidbs.schema.Table(table_name=table.llm_table_name,
                                                        file_name=str(table.llm_table_name) + ".csv",
                                                        columns=columns_per_table[table.llm_table_name],
                                                        foreign_keys=foreign_keys)
                    final_tables.append(schema_table)


                    ## save csv files (with and without qids)
                    filename = table.llm_table_name + ".csv"
                    table_df_save = table.llm_renamed_df.map(lambda x: x[0] if isinstance(x, list) else x)
                    table_df_save.columns = table_df_save.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
                    table_df_save.to_csv(db_path / "tables" / filename, index=False)

                    ## prepare final df with qids
                    table_df_save =  table.llm_renamed_df.map(lambda x: (x[0], x[1]) if isinstance(x, list) else x)
                    table_df_save.columns = table_df_save.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
                    table_df_save.to_csv(db_path / "tables_with_item_ids" / filename, index=False)

                start_table = database.tables[0]
                ### Create schema (needs database name and tables)
                database_name = database.db_name
                schema = wikidbs.schema.Schema(database_name=database_name,
                                            wikidata_property_id=start_table.predicate["id"],
                                            wikidata_property_label=start_table.predicate["label"],
                                            wikidata_topic_item_id="Q"+start_table.object["id"],
                                            wikidata_topic_item_label=start_table.object["label"],
                                            tables=final_tables)

                ###########################
                # serialize schema to disk 
                ###########################
                with open(db_path / "schema.json", "w", encoding="utf-8") as file:
                    json.dump(converter.unstructure(schema), file, indent=2)

                with open(db_path / "schema.json", "r", encoding="utf-8") as file:
                    schema_test_load = converter.structure(json.load(file), wikidbs.schema.Schema)


                # visualize the schema
                create_schema_diagram(tables=final_tables, save_path=db_path, show_diagram=False)

                with open(db_path / "database.json", "w", encoding="utf-8") as file:
                    for table in database.tables:
                        table.rows = None  # cannot serialize rows since dict keys are tuples...
                        table.full_properties_with_outgoing_items = None  # cannot serialize...
                        table.properties_with_outgoing_items = None  # cannot serialize...
                    json.dump(converter.unstructure(database), file)

                with open(db_path / "database.json", "r", encoding="utf-8") as file:
                    database_test_load = converter.structure(json.load(file), Database)

                statistics["number_of_tables"].append(len(database.tables))
                statistics["number_of_cols"].append([len(table.table_df.columns) for table in database.tables])
                statistics["number_of_rows"].append([len(table.table_df) for table in database.tables])
                statistics["table_names_orig"].append([table.table_name for table in database.tables])
                statistics["table_names_llm"].append([table.llm_table_name for table in database.tables])   

                # intermediate saving of statistics
                with open(creation_run_path / "db_statistics.json", "w", encoding="utf-8") as statistics_file:
                    json.dump(statistics, statistics_file, indent=2)          


            if cfg.limit and num_created_databases >= cfg.limit:
                break
        
        log.info(f"Created DBs: {num_created_databases}")


if __name__=="__main__":
    start_db_creation()
   
   