import logging
import pandas as pd
from pathlib import Path

from wikidbs.utils import differentiate_datatype
from wikidbs.table import Table

log = logging.getLogger(__name__)

non_suitable_relations = {"P1889": "different from",
                          "P460": "said to be the same as",
                          "P31": "instance of",
                          "P279": "subclass of",
                          "P910": "topic's main category",
                          "P527": "has_part(s)",
                          "P1424": "topic's main template",
                          "P5008": "on focus list of Wikimedia project",
                          "P8989": "category for the view of the item",
                          "P2354": "has list", # may be useful in future?
                          "P6104": "maintained by WikiProject",
                          "P2888": "exact match",
                          "P5869": "model item",
                          "P7867": "category for maps",
                          "P1151": "topic's main Wikimedia portal",
                          "P2687": "Wikidata property",
                          "P5125": "Wikimedia outline",
                          "P155": "follows",
                          "P156": "followed by",
                          "P1687": "Wikidata property",
                          "P1365": "replaces",
                          "P1366": "replaced by"
                        }   
IS_WIKIDATA_ID = r"Q\d+"

def build_table_rows(table_items: dict, properties_lookup_df: pd.DataFrame, wikidata_labels: dict):
    """
    Transform the given wikidata items for the table into rows
    """
    table_rows = []
    properties_with_outgoing_items = set()

    for wikidata_item in table_items:
        current_row = {}

        # add label and description of the item
        current_row[("label", "wikidata_id", "datatype")] = (wikidata_item["label"], wikidata_item["wikidataId"], "string")
        current_row[("description", None, "datatype")] = (wikidata_item['description'], None, "string")

        # add all properties of the item as columns
        for property_id, property_info in wikidata_item["properties"].items():
            is_coordinate = False
            if property_info == {}:
                continue

            # replace each property with its natural language label:
            try:
                column_label = properties_lookup_df[properties_lookup_df["wikidataId"] == property_id]["label"].item()
            except:
                column_label = "unknown"
                log.error(f"Did not find column label of property: {property_id}")
            
            try:
                datatype = property_info["dataType"]
            except:
                log.error(f"Couldn't find datatype in property {property_id}: *{property_info}*")
            column_def = (column_label, property_id, "datatype")

            # get the cell value
            if datatype == "wikibase-entityid":
                try:
                    q_id = property_info["value"]["numeric-id"]
                except KeyError as e:
                    if property_info["entityType"] == "wikibase-form":
                        continue
                    if property_info["entityType"] == "wikibase-sense":
                        continue
                    print(e)
                    print(current_row)
                    print(property_info)
                    raise KeyError
                
                #  get natural language label of Q-ID 
                try: 
                    cell_value_label = wikidata_labels[str(q_id)]
                except:
                    cell_value_label = "" 

                cell_value_q_id = "Q" + str(q_id)
                
                # collect possible outgoing links
                if column_def not in properties_with_outgoing_items and property_id not in non_suitable_relations and "category for" not in column_label:
                    properties_with_outgoing_items.add(column_def)
            elif datatype == "string":
                cell_value_label = property_info["value"]
                cell_value_q_id = None
            else: # final value, not wikidata item
                if datatype == "globecoordinate":
                    is_coordinate = True
                    cell_value_q_id = None
                    # need to add one column for latitude and one for longitude
                    for l in ["latitude", "longitude"]:
                        cell_value_label = property_info["value"][l]
                        new_column_def = (f"{column_def[0]}_{l}", column_def[1], column_def[2])
                        cell_value = (cell_value_label, cell_value_q_id, datatype)
                        current_row[new_column_def] = cell_value
                else:
                    datatype_keyword = differentiate_datatype(datatype=datatype)
                    cell_value_label = property_info["value"][datatype_keyword]                
                cell_value_q_id = None

            if not is_coordinate:
                if datatype == "wikibase-entityid":
                    cell_value = [cell_value_label, cell_value_q_id, datatype] # intermediately saving as list, later converting to tuple
                else:
                    cell_value = (cell_value_label, cell_value_q_id, datatype)
                current_row[column_def] = cell_value

        table_rows.append(current_row)

    return table_rows, properties_with_outgoing_items