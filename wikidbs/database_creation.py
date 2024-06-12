from omegaconf import DictConfig

import logging
import pandas as pd
import random

from sentence_transformers import SentenceTransformer

from wikidbs.table_creation import build_table_rows
from wikidbs.table import Table
from wikidbs.database import Database
from wikidbs.connected_table_creation import create_connected_table

log = logging.getLogger(__name__)

def create_database_for_topic(cfg: DictConfig, db, topic_dict: dict, properties_lookup_df: pd.DataFrame, embedding_model: SentenceTransformer, wikidata_labels: dict, wikidata_properties):
    """
    Create a database for the given topic
    """
    ###########################################
    # Create the start table
    ###########################################
    current_threshold = cfg.embedding_similarity_threshold
    entity_ids_of_rows = topic_dict["subjects"]

    # query table items for the main table based on the ids from topic row list

    log.debug(f"Querying data from db for {len(entity_ids_of_rows)} items")
    row_entity_information = list(db.items.find({"_id": {"$in": entity_ids_of_rows}})) # once per database

    # create a table for the given topic
    table_rows, properties_with_outgoing_items = build_table_rows(table_items=row_entity_information, properties_lookup_df=properties_lookup_df, wikidata_labels=wikidata_labels)
    
    table_name = topic_dict["predicate_label"] + " " + topic_dict["object_label"]
    table_name = table_name.replace("/", " ")


    start_table = Table.create(table_name=table_name,
                        predicate_id=topic_dict["predicate"],
                        predicate=topic_dict["predicate_label"],
                        object=topic_dict["object_label"],
                        object_id=topic_dict["object"],
                        row_entity_ids=row_entity_information,
                        rows=table_rows,
                        properties_with_outgoing_items=properties_with_outgoing_items,
                        )
    
    start_table.transform_to_dataframe(cfg.max_sparsity_of_column)
    start_table.find_outgoing_relations(cfg.min_rows, embedding_model=embedding_model)

    if len(start_table.possible_relations) == 0:
        log.error(f"No outgoing relations from main table, can't build database")
        return None

    ###########################################
    # Initialize the database
    ###########################################
    database = Database.from_start_table(start_table=start_table)
    assert database.start_table

    database.initialize_semantic_embedding(embedding_model=embedding_model)

    ###########################################
    # Create connected tables
    ###########################################

    # while there are still relations: 
    while len(database.further_relations) > 0:
        # check if table limit is already reached
        if cfg.max_num_tables and len(database.tables) >= cfg.max_num_tables:
            break

        ############################
        # choose the next relation:  
        ############################
        # check if there is a relation over the similarity threshold to choose next:
        chosen_relation, similarity_score = database.find_semantically_most_similar_relation(embedding_model=embedding_model)

        if similarity_score < float(current_threshold):
            if len(database.tables) < 2:
                current_threshold=0.05
                log.debug("Reducing similarity score")
            if current_threshold<cfg.embedding_similarity_threshold and len(database.tables) > 3:
                current_threshold = cfg.embedding_similarity_threshold
            else:
                #print(f"Not taking: {[(x.property_label, x.current_similarity) for x in database.further_relations]}")
                log.debug("No further relations are suitable")
                break
        
        log.debug(f"Relation: *{chosen_relation.property_label}* from *{chosen_relation.parent_table}* table with similarity_score {similarity_score}")

        ##########################
        # create connected table
        ##########################
        parent_topic_label = chosen_relation.parent_table
        connected_table = create_connected_table(db=db, database=database, 
                                                 properties_lookup_df=properties_lookup_df, 
                                                 chosen_relation=chosen_relation, 
                                                 parent_topic_label=parent_topic_label, 
                                                 cfg=cfg,
                                                 embedding_model=embedding_model,
                                                 similarity_score=similarity_score,
                                                 wikidata_labels=wikidata_labels,
                                                 wikidata_properties=wikidata_properties)

        # if rows are added to an existing table, no additional table is created
        if connected_table:
            database.tables.append(connected_table)
            database.further_relations += connected_table.possible_relations

            # update the database embedding:
            connected_table_embedding = embedding_model.encode(connected_table.table_name, convert_to_tensor=True, show_progress_bar=False)
            database.semantic_embedding = (cfg.embedding_weight_updates * database.semantic_embedding) + ((1-cfg.embedding_weight_updates) * connected_table_embedding)

    if len(database.tables) < 2:
        return None

    return database