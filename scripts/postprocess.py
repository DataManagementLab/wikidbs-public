import json
import logging
import math
import multiprocessing
import random
import re
import time
import ast
from pathlib import Path
import unicodedata

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from wikidbs import utils
import wikidbs.schema
from wikidbs.database import Database
from wikidbs.serialization import converter
from wikidbs.visualize import create_schema_diagram

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="postprocess.yaml")
def start_postprocessing(cfg: DictConfig):
    # create output folder if necessary
    output_path = Path(f"{cfg.output_folder}")
    output_path.mkdir(parents=True, exist_ok=True)

    dbs_to_process = [x for x in list(Path(cfg.input_folder).iterdir()) if x.is_dir()]
    dbs_to_process = dbs_to_process[:cfg.limit]

    log.info(f"Starting to post-process databases from {cfg.input_folder}, limit is {cfg.limit}, plan to rename {(len(dbs_to_process))}")


    before = time.time()
    if cfg.processes is None:

        for db_to_process in tqdm(dbs_to_process, "handle lines"):
            handle_line(db_to_process, output_path, cfg)
    else:
        line_group_size = math.ceil(len(dbs_to_process) / cfg.processes)
        line_groups = []
        for left in range(0, len(dbs_to_process), line_group_size):
            line_groups.append(dbs_to_process[left:left + line_group_size])
        all_params = [(line_group, output_path, cfg) for line_group in line_groups]

        with multiprocessing.Pool(cfg.processes) as pool:
            for _ in pool.imap(wrapper, all_params):
                pass
    log.info(f"Done! Processed DBs in {time.time() - before:.2f} seconds.")


def handle_line(
        db_folder: Path,
        output_path: Path,
        cfg: DictConfig
) -> dict | None:
   
    # check if database has already been renamed
    if output_path / db_folder.name in list(output_path.iterdir()):
        log.info(f"DB {db_folder.name} has already be post-processed")
        return

    try: 
        log.info(f"Database: *{db_folder.name}")

        assert Path(db_folder / "database.json").exists()

        # load database
        with open(db_folder / "database.json", "r", encoding="utf-8") as file:
            database_to_process = converter.structure(json.load(file), Database)
        # convert dataframe tuples back to real tuples:
        for table in database_to_process.tables:
            table.table_df.columns = [ast.literal_eval(col) for col in table.table_df.columns]

        # create schema information
        final_tables = []
        col_names_first_col = {}
        columns_per_table = {}

        for table_idx, table in enumerate(database_to_process.tables):
            orig_table = database_to_process.tables[table_idx]

            # Determine the datatype for each column
            column_types = {col: utils.majority_type(orig_table.table_df[col]) for col in orig_table.table_df.columns}
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

        Path(db_folder / "tables").mkdir()
        Path(db_folder / "tables_with_item_ids").mkdir()

        for table_idx, table in enumerate(database_to_process.tables):
            # handle foreign keys

            foreign_keys = []
            for fk in table.foreign_keys:
                fk_col = fk.column_name

                fk_reference_table = fk.reference_table_name
                reference_col = col_names_first_col[fk_reference_table]
                schema_fk = wikidbs.schema.ForeignKey(column=fk_col,
                                                    reference_column=reference_col,
                                                    reference_table=fk_reference_table)
                foreign_keys.append(schema_fk)


            #print(f"Foreign keys are: {foreign_keys}")
            schema_table = wikidbs.schema.Table(table_name=table.llm_table_name,
                                                file_name=str(table.llm_table_name) + ".csv",
                                                columns=columns_per_table[table.llm_table_name],
                                                foreign_keys=foreign_keys)
            final_tables.append(schema_table)


            ## save csv files (with and without qids)
            filename = table.llm_table_name + ".csv"
            table_df_save = table.llm_renamed_df.map(lambda x: x[0] if isinstance(x, list) else x)
            table_df_save.columns = table_df_save.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
            table_df_save.to_csv(db_folder / "tables" / filename, index=False)

            ## prepare final df with qids
            orig_table = database_to_process.tables[table_idx]
            table_df_save = orig_table.table_df.map(lambda x: (x[0], x[1]) if isinstance(x, list) else x)
            table_df_save.columns = table.columns #cols_with_pid

            table_df_save.to_csv(db_folder / "tables_with_item_ids" / filename, index=False)

        start_table = database_to_process.tables[0]
        ### Create schema (needs database name and tables)
        database_name = database_to_process.db_name
        schema = wikidbs.schema.Schema(database_name=database_name,
                                    wikidata_property_id=start_table.predicate["id"],
                                    wikidata_property_label=start_table.predicate["label"],
                                    wikidata_topic_item_id="Q"+start_table.object["id"],
                                    wikidata_topic_item_label=start_table.object["label"],
                                    tables=final_tables)

        ###########################
        # serialize schema to disk 
        ###########################
        with open(db_folder / "schema.json", "w", encoding="utf-8") as file:
            json.dump(converter.unstructure(schema), file, indent=2)

        with open(db_folder / "schema.json", "r", encoding="utf-8") as file:
            schema_test_load = converter.structure(json.load(file), wikidbs.schema.Schema)


        # visualize the schema
        create_schema_diagram(tables=final_tables, save_path=db_folder, show_diagram=False)

        with open(db_folder / "database.json", "w", encoding="utf-8") as file:
            for table in database_to_process.tables:
                table.rows = None  # cannot serialize rows since dict keys are tuples...
                table.full_properties_with_outgoing_items = None  # cannot serialize...
                table.properties_with_outgoing_items = None  # cannot serialize...
            json.dump(converter.unstructure(database_to_process), file)

        with open(db_folder / "database.json", "r", encoding="utf-8") as file:
            database_test_load = converter.structure(json.load(file), Database)

    except Exception as e:
        log.warning(f"Failed to post-process database {db_folder.name}", exc_info=True)

        if isinstance(e, NotImplementedError):
            raise NotImplementedError(e)

    return 

# EXECUTE ONCE PER RUN
postprocess_names_random = random.Random(469866043)
postprocess_name_modes = [
    # "no_lowercase",  # 'countryname'
    # "no_uppercase",  # 'COUNTRYNAME'
    "no_pascal",  # 'CountryName'
    "no_pascal",  # 'CountryName'
    "no_pascal",  # 'CountryName'
    "spaces_lowercase",  # 'country name'
    "spaces_uppercase",  # 'COUNTRY NAME'
    "spaces_pascal",  # 'Country Name'
    "underscores_lowercase",  # 'country_name'
    "underscores_uppercase",  # 'COUNTRY_NAME'
    "underscores_pascal",  # 'Country_Name'
    "hyphen_lowercase",  # 'country-name'
    "hyphen_uppercase",  # 'COUNTRY-NAME'
    "hyphen_pascal",  # 'Country-Name'
]

def is_camel_case_naive(s: str):
  is_camel = True
  if s[1:] == s[1:].lower():
    is_camel = False
  if s[1:] == s[1:].upper():
    is_camel = False
  if " " in s:
    is_camel = False
  if "_" in s:
    is_camel = False
  return is_camel

def postprocess_name(name: str, mode: str) -> str:
    # do not allow unicode, taken from https://github.com/django/django/blob/master/django/utils/text.py
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s-]", "", name)

    # split into parts
    name = re.sub(r"[-\s]+", "-", name).strip("-_")
    parts = name.split("-")
    parts = [p for part in parts for p in part.split("_")]
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


def handle_line_group(
        line_group,
        output_path: Path,
        cfg: DictConfig
) -> list[dict | None]:
    
    return [handle_line(line, output_path, cfg) for line in line_group]


def wrapper(args):
    return handle_line_group(*args)


if __name__ == "__main__":
    start_postprocessing()
