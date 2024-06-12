import re
import ast
import pathlib

import numpy as np
from omegaconf import DictConfig, OmegaConf

from wikidbs.database import Database
from wikidbs.table import Table, ForeignKey
from wikidbs.llm_utils import fill_chat_template, execute_requests_against_api, extract_text_from_response
from wikidbs.utils import find_duplicates_positions

# Regular expression pattern to match a list in the string
_LIST_PATTERN = re.compile(r'\[(.*?)\]')
_sample_rows_random = np.random.default_rng(seed=351338462)


def rename(cfg: DictConfig, database: Database, responses_dir: pathlib.Path, history_semaphore, history):
    """
    Renames the tables and columns of all tables using a language model.
    """
    # compute costs and prompt user to agree 
    # total_max_cost = len(database.tables) * 0.026  # approximate current costs using GPT-4, need one request per table
    # input(f"Press enter to continue and spend up to around ${total_max_cost:.4f}.")

    # first rename the start table of the database
    rename_start_table(cfg=cfg, database=database, responses_dir=responses_dir, history_semaphore=history_semaphore, history=history)

    # rename all the other tables
    for table_to_rename in database.tables[1:]:
        rename_connected_table(cfg=cfg, 
                               database=database, 
                               table_to_rename=table_to_rename, 
                               foreign_keys=database.foreign_keys, 
                               responses_dir=responses_dir, history_semaphore=history_semaphore, history=history)


def prepare_prompt_table(cfg: DictConfig, table: Table):
    """
    TODO: document
    """
    # prepare table for request, only include columns that don't contain IDs of something
    cols_to_change_name_of = [x for x in table.columns if not "ID" in x[0]]
    try:
        start_table_df_for_request = table.table_df[cols_to_change_name_of].sample(n=cfg.num_table_rows_in_prompt, random_state=_sample_rows_random, ignore_index=True)
    except:
        print(f"Table rows: {len(table.table_df[cols_to_change_name_of])}, {cols_to_change_name_of} len cols to change name of: {len(cols_to_change_name_of)}")
    start_table_df_for_request = start_table_df_for_request.map(lambda x: x[0] if isinstance(x, tuple) else x)
    start_table_df_for_request = start_table_df_for_request.rename(columns={x: x[0] for x in start_table_df_for_request.columns})
    if cfg.serialization == "markdown":
        start_table_linearized = start_table_df_for_request.to_markdown(index=False)
    elif cfg.serialization == "csv":
        start_table_linearized = start_table_df_for_request.to_csv(index=False)
    else:
        raise ValueError(f"Unsupported serialization format: {cfg.serialization}")
    return start_table_linearized, cols_to_change_name_of

def parse_new_column_names(response_part: str, table: Table, cols_to_change_name_of: list):
    """
    Tries to convert the LLM response containing the new column names to a python list and renames the table dataframe.
    """
    # convert llm response string to a python list
    try: 
        # first parsing attempt
        new_col_names_list = ast.literal_eval(response_part)
    except:
        # check if response contains brackets [
        if "[" not in response_part:
            elements = [element.strip() for element in response_part.split(',')]

            elements_with_quotes = []
            for element in elements:
                if not ((element.startswith("'") and element.endswith("'")) or 
                        (element.startswith('"') and element.endswith('"'))):
                    elements_with_quotes.append(f"'{element}'")
                else:
                    elements_with_quotes.append(element)

            # Join the elements back into a string representation of a list
            response_part = f"[{', '.join(elements_with_quotes)}]"

            #print(f"Now trying to parse: *{response_part}*")
        else:
            # try to remove text that is in front or after the list
            # Search for the list pattern in the input string
            match = _LIST_PATTERN.search(response_part)
            
            # Extract the list portion of the string
            if match:
                response_part = match.group(0)
            
        try:
            # second parsing attempt
            new_col_names_list = ast.literal_eval(response_part)
        except Exception as e: 
            print(f"Could not parse new col names: *{repr(response_part)}*")
            print(e)
            # need to keep the old names
            new_col_names_list = []
    
    renamed_columns = {}
    try:
        # build renamed dataframe:
        assert len(cols_to_change_name_of) == len(new_col_names_list)

        duplicate_positions = []
        # check if there are duplicates in the new column names:
        if len(new_col_names_list) != len(set(new_col_names_list)):
            print("Duplicates in column names..")
            duplicate_positions = find_duplicates_positions(new_col_names_list)

        for orig_col in table.table_df.columns:
            if orig_col in cols_to_change_name_of:
                col_idx = cols_to_change_name_of.index(orig_col)
                llm_col_name = str(new_col_names_list[col_idx])
                if llm_col_name != "" and col_idx not in duplicate_positions:
                    renamed_columns[orig_col] = (llm_col_name, orig_col[1], orig_col[2])
                else:
                    renamed_columns[orig_col] = orig_col
            else:
                # just keep the old name
                renamed_columns[orig_col] = orig_col
    except AssertionError:
        print(f"LLM predicted col names have the wrong length: {new_col_names_list} with len {len(new_col_names_list)} instead of {len(cols_to_change_name_of)}, need to keep the originial names")
        for orig_col in table.table_df.columns:
            # just keep the old name
            renamed_columns[orig_col] = orig_col

    renamed_table_df = table.table_df.copy()
    renamed_table_df = renamed_table_df.rename(columns=renamed_columns)
    table.llm_renamed_df = renamed_table_df

def rename_start_table(cfg: DictConfig, database: Database, responses_dir: pathlib.Path, history_semaphore, history):
    request = {
        "model": cfg.model,
        #"max_tokens": max_tokens_for_ground_truth(ground_truth, cfg.api_name, cfg.model, cfg.max_tokens_over_ground_truth),
        "temperature": cfg.temperature
    }

    start_table_linearized, cols_to_change_name_of = prepare_prompt_table(cfg=cfg, table=database.start_table)

    request["messages"] = fill_chat_template(
        OmegaConf.to_container(cfg.prompt_template_start),
        database_start_table_name=database.start_table.table_name,
        start_table = start_table_linearized
    )

    responses = execute_requests_against_api(requests=[request], api_name=cfg.api_name, responses_dir=responses_dir, history_semaphore=history_semaphore, history=history)
    response_text = extract_text_from_response(responses[0])
    if response_text == None:
        # an error ocurred
        error_type = responses[0]["error"]["type"]
        if error_type == "insufficient_quota":
            raise RuntimeError(f"Cost limit of project reached.")
        else:
            raise NotImplementedError(f"Error occured: {responses[0]['error']}")

    response_splitted = response_text.split("\n")
    response_splitted = [r for r in response_splitted if r.strip() != ""]

    # new name for the database
    llm_database_name = response_splitted[0].replace("1.", "").strip()
    print(f"For DB *{database.start_table.table_name}*")
    print(f"LLM database name: *{llm_database_name}*")
    database.db_name = llm_database_name

    llm_start_table_name = response_splitted[1].replace("2.", "").strip()
    print(f"LLM start table name: *{llm_start_table_name}*")
    database.start_table.llm_table_name = llm_start_table_name


    new_col_names_part = response_splitted[2].replace("3.", "").strip()
    parse_new_column_names(response_part=new_col_names_part,
                           table=database.start_table,
                           cols_to_change_name_of=cols_to_change_name_of)

    

def rename_connected_table(cfg: DictConfig, database: Database, table_to_rename: Table, foreign_keys: list[ForeignKey], responses_dir: pathlib.Path, history_semaphore, history):
    request = {
        "model": cfg.model,
        #"max_tokens": max_tokens_for_ground_truth(ground_truth, cfg.api_name, cfg.model, cfg.max_tokens_over_ground_truth),
        "temperature": cfg.temperature
    }

    fk_table_linearized, cols_to_change_name_of = prepare_prompt_table(cfg=cfg, table=table_to_rename)

    # prepare foreign key relationships to the given table (strings parent_table.column_name)
    table_foreign_keys = []
    table_foreign_key_columns = []
    for fk in foreign_keys:
        if fk.reference_table_name == table_to_rename.table_name:
            table_foreign_keys.append(f"{fk.source_table_name}.{fk.column_name}")
            table_foreign_key_columns.append(str(fk.column_name))

    request["messages"] = fill_chat_template(
        OmegaConf.to_container(cfg.prompt_template_fks),
        database_start_table_name = database.start_table.table_name,
        database_name = database.db_name,
        fk_table = fk_table_linearized,
        fk_table_name = table_to_rename.table_name,
        fk_relationships = str(table_foreign_keys),
        fk_columns = str(table_foreign_key_columns)
    )

    responses = execute_requests_against_api(requests=[request], api_name=cfg.api_name, responses_dir=responses_dir, history_semaphore=history_semaphore, history=history)
    response_text = extract_text_from_response(responses[0])
    response_splitted = response_text.split("\n")
    response_splitted = [r for r in response_splitted if r.strip() != ""]

    # new table name
    llm_table_name = response_splitted[0].replace("1.", "").strip()
    print(f"LLM table name: *{llm_table_name}* instead of *{table_to_rename.table_name}*")
    table_to_rename.llm_table_name = llm_table_name

    new_col_names_part = response_splitted[1].replace("2.", "").strip()
    parse_new_column_names(response_part=new_col_names_part,
                        table=table_to_rename,
                        cols_to_change_name_of=cols_to_change_name_of)
