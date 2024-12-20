import re
import unicodedata 
from pathlib import Path
import logging
import random
import pandas as pd

log = logging.getLogger(__name__)

from collections import Counter, defaultdict

from wikidbs.table import Table, ForeignKey

def differentiate_datatype(datatype: str):
    if datatype == "string":
        return "value"
    elif datatype == "quantity":
        return "amount"
    elif datatype == "monolingualtext":
        return "text"
    elif datatype == "globecoordinate":
        raise ValueError(f"Differentiate datatype should not be called with globecoordinate, latitude and longitude need to be handled seperately.")
    elif datatype == "time":
        return "time"
    else:
        log.error(f"Error: encountered new datatype: *{datatype}*")
        return None

def save_dataframe_as_csv_file(table_df, save_path, topic_label: str, parent_table: str=None, parent_relation: str=None):
    """
    Saves the dataframe to the filesystem as csv file, filters out descriptions and optionally also qids.
    The filename structure is: 'topicLabel___parentTable--parentRelation_numRows.csv'

        Args:
            table_df (Dataframe): A pandas dataframe to be written to the filesystem
            save_path (Path): The folder path to save the dataframe to
            topic_label (str): The topic of the dataframe (table name)
            parent_table (str): The topic of the parent table 
            parent_relation (str): The property of the foreign key in the parent table

        Returns:
            path: The path where the dataframe was written to
    """
    # do not save description:
    table_df_save = table_df.copy()
    table_df_save_ids = table_df.copy()
    table_df_save = table_df_save.map(lambda x: x[0] if type(x) == tuple else x)
    table_df_save_with_ids = table_df_save_ids.map(lambda x: [x[0], x[1]] if type(x) == tuple else x)
    # build filename
    #filename = slugify(topic_label) + ".csv"
    filename = topic_label + ".csv"
    # save dataframe
    table_df_save.to_csv(save_path / "tables" / Path(filename), index=False)
    table_df_save_with_ids.to_csv(save_path / "tables_with_ids" / Path(filename), index=False)
    return Path(filename)

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    #assert not "___" in value, print(value)
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def find__col_idx_of_fk_column(table, foreign_key):
    """
    Find index of a renamed column in the orignial (non-renamed) dataframe
    """
    for col_idx, col_triple in enumerate(list(table.table_df.columns)):
        if col_triple[0] == foreign_key.column_name:
            return col_idx
    # not found?
    raise ValueError(f"Column {foreign_key.column_name} not found in {table.table_df.columns}")

def find_duplicates_positions(lst):
    positions = defaultdict(list)

    # Populate the dictionary with positions
    for index, item in enumerate(lst):
        positions[item.lower()].append(index)

    # Extract only the items that have duplicates (more than one position)
    duplicate_positions = []
    for _, pos in positions.items():
        if len(pos) > 1:
            duplicate_positions += pos

    return duplicate_positions


def majority_type(column_values):
    # Extract datatypes from the tuples
    types = []
    for col in column_values:
        # col might be nan
        if isinstance(col, tuple) or isinstance(col, list):
            col_type = col[2]
            if col_type is not None:
                types.append(col_type)
    # Count the occurrences of each type
    type_count = Counter(types)
    # Get the most common type
    most_common_type = type_count.most_common(1)[0][0]
    return most_common_type