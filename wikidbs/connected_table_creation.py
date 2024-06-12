import pandas as pd
import logging

from wikidbs.table import Table, ForeignKey
from wikidbs.database import Database
from wikidbs.table_creation import build_table_rows

log = logging.getLogger(__name__)

def create_connected_table(db, database: Database, properties_lookup_df: pd.DataFrame, chosen_relation, parent_topic_label, cfg, embedding_model, similarity_score: float, wikidata_labels: dict, wikidata_properties: dict):
    next_table_name = chosen_relation.property_label
    next_table_name = next_table_name.replace("/", "-")
    property_info = wikidata_properties[chosen_relation.property_id[1:]]

    # try to find a better name for the connected table than the relation name
    alternative_name = try_to_find_alternative_name_for_relation(db=db, property_info=property_info)
    if alternative_name is not None:
        next_table_name = alternative_name
        log.debug(f"Replaced Name! Old name: *{chosen_relation.property_label}* New name: *{next_table_name}*")

    # check if there is already a table for the chosen relation in the database
    table_already_existent = False
    if next_table_name in database.get_table_names():
        existent_table = database.get_table(table_name=next_table_name)
        existent_df = existent_table.table_df
        log.debug(f"existing table {next_table_name} found with {len(existent_df)} rows")
        existent_df_values = list(existent_df[('label', 'wikidata_id', 'datatype')])
        table_already_existent = True

    # take column values from parent dataframe
    item_ids_to_fetch = []

    for x in chosen_relation.distinct_entities.items():
        item_label, item_qid, _ = x[1]
        # skip empty lines
        if item_label == "":
            continue
        # only add item if it is not existent in previous df
        if not table_already_existent or not item_label in existent_df_values:
            item_ids_to_fetch.append(int(item_qid[1:]))

    if len(item_ids_to_fetch) > 0:
        log.debug(f"Querying data from db for {len(item_ids_to_fetch)} items")
        row_entity_information = list(db.items.find({"_id": {"$in": item_ids_to_fetch}})) # one request for each connected table to get all row information
    else:
        row_entity_information = []
        log.debug(f"No additional data for the existing table")

    #log.info(f"### Connected table: *{parent_topic_label} - {chosen_relation.property_label} ({chosen_relation.property_id})* ----> *{next_table_name}*")
    connected_table = None

    # create next table
    if len(row_entity_information) > 0:
        table_rows, next_properties_with_outgoing_items = build_table_rows(table_items=row_entity_information, properties_lookup_df=properties_lookup_df, wikidata_labels=wikidata_labels)

        # if not already existent table, create a new one:
        if not table_already_existent:
            connected_table = Table.create(table_name=next_table_name,
                                row_entity_ids=row_entity_information,
                                rows=table_rows,
                                properties_with_outgoing_items=next_properties_with_outgoing_items,
                                )

            connected_table.transform_to_dataframe(cfg.max_sparsity_of_column)
            connected_table.find_outgoing_relations(cfg.min_rows, embedding_model=embedding_model)
        else:
            existent_table.rows.append(table_rows) # append rows
            existent_table.row_entity_ids.append(row_entity_information)

            # only use the columns from the already existing table
            additional_rows_df = pd.DataFrame( [{key: value for key, value in current_row.items()} for current_row in table_rows])
            columns_existent = [x for x in existent_df.columns if x in additional_rows_df.columns]
            additional_rows_df = additional_rows_df[columns_existent]

            # concatenate the two dataframes
            updated_df = pd.concat([existent_df, additional_rows_df], ignore_index=True)

            existent_table.table_df = updated_df

            log.debug(f"Added {len(table_rows)} more rows to {existent_table.table_name} ")

    foreign_key = ForeignKey(source_table_name=parent_topic_label,
                            property_id=chosen_relation.property_id,
                            column_name=chosen_relation.property_label,
                            reference_table_name=next_table_name,
                            similarity=similarity_score)
    
    source_table = database.get_table(table_name=parent_topic_label)
    source_table.foreign_keys.append(foreign_key)
    database.foreign_keys.append(foreign_key)
    
    return connected_table

def try_to_find_alternative_name_for_relation(db, property_info):
    """
    Makes use of the "Wikidata item of this property" (=P1629) Property to find a better name for the next table
    """
    try:
        # P1629 = "Wikidata item of this property"
        alternative_names = property_info["properties"]["P1629"]["value"]
        assert type(alternative_names) == dict, print("Type is:", type(alternative_names))
        alternative_name_id = alternative_names["numeric-id"]
        # get label of QID from db.items
        alternative_name_label = list(db.items.find({"_id": int(alternative_name_id)}))[0]["label"] # one request per connected table
        alternative_name_label = alternative_name_label.replace("/", "-")
        return alternative_name_label
    except:
        #print("No *wikidata item of this property* property")
        return None

