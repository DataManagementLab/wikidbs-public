import json
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import logging

from wikidbs.mongodb import get_db_connection

log = logging.getLogger(__name__)

def convert_profiling_dict_to_jsonlines(cfg: DictConfig):
    """
    Saves the converted profiling dict in data/profiling and returns a list with statistics and a list with the numbers of rows
    """
    log.info(f"Starting to convert profiling dict *{cfg.profiling_dict}* with limit: *{cfg.limit}*")
    num_rows = []

    profiling_dict_path = Path(cfg.profiling_dict)

    topics_info = []

    if cfg.get_nl_labels:
        db = get_db_connection(cfg)
        # get property lookup dict
        properties = db.properties.find({}, {"label": 1, "wikidataId": 1, "_id": 0})
        property_lookup_df = pd.DataFrame(properties)

    # New structure:
    ## {"predicate": "Px", "object": "xyz", "num_rows": n, "subjects": ["a", "b", ....]}
    converted_entities = 0

    with open(profiling_dict_path, "r") as input_file, open(Path(f"data/profiling/{cfg.converted_dict_name}"), "w") as output_file:
        log.info("Now loading profiling dict into RAM")
        data = json.load(input_file)
        log.info("Done loading profiling dict into RAM")

        for predicate in tqdm(data.keys()):
            for object in data[predicate].keys():
                num_rows = len(data[predicate][object])
                entity_line_dict = {"predicate": predicate, "object": object, "num_rows": num_rows, "subjects": data[predicate][object]}
                num_rows.append(num_rows)

                if cfg.get_nl_labels:
                    if not cfg.label_names_min_num_rows or num_rows >= cfg.label_names_min_num_rows:
                        predicate_label, object_label = get_natural_language_labels(db, entity_dict=entity_line_dict, property_lookup_df=property_lookup_df)
                        topics_info.append((predicate, predicate_label, object, object_label, num_rows))
                        entity_line_dict["predicate_label"] = predicate_label
                        entity_line_dict["object_label"] = object_label

                output_file.write(json.dumps(entity_line_dict) + '\n')

                converted_entities += 1
                if cfg.limit and converted_entities >= cfg.limit:
                    break
            if cfg.limit and converted_entities >= cfg.limit:
                    break

    if cfg.get_nl_labels:
        # convert table_info to dataframe
        table_info_df = pd.DataFrame(topics_info, columns = ["PredicateID", "PredicateLabel", "ObjectID", "ObjectLabel", "NumRows"])
        # write topic df to disk
        table_info_df.to_csv(f"./data/profiling/possible_topics_{len(table_info_df)}.csv")

    
    log.info(f"Done with profiling, converted {converted_entities} entities")

    return num_rows


def get_natural_language_labels(db, entity_dict: dict, property_lookup_df: pd.DataFrame):
    # get natural language label of predicate
    predicate_label = property_lookup_df[property_lookup_df["wikidataId"] == entity_dict["predicate"]]["label"].item()

    # get natural label of object
    try:
        object_label = list(db.items.find({"_id": int(entity_dict["object"])}))[0]["label"]
    except IndexError:
        object_label = ""

    object_label = object_label.replace(",", ";")

    return predicate_label, object_label