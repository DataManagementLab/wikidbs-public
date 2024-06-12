import os
import gzip
import json
from tqdm import tqdm
import logging
from pathlib import Path

log = logging.getLogger(__name__)

def preprocess_wikidata_dump(cfg, db):
    log.info(f"Starting to process wikidata dump, limit is {cfg.limit}")
    itemsCollection = db["items"]
    propertiesCollection = db["properties"]

    concept_profiling = {} # keys are properties, values are dictionaries, e.g. {object1: [subject1, subject2] }
    concept_profiling_checkpoints = []

    no_english_label_counter = 0
    num_entities_processed = 0

    dump_path = Path(cfg.wikidata_dump)

    # loop over gz zipped wikidata dump
    with gzip.open(dump_path, "rb") as file:
        for line_number, line in enumerate(tqdm(file)):
            if cfg.limit and line_number > cfg.limit:
                    log.info(f"Reached limit of {cfg.limit}")
                    break

            decoded_line = line.decode("UTF-8")
            # dismiss the ,/n at the end of each line
            try:
                if decoded_line[-2] == ",":
                    json_line = json.loads(decoded_line[:-2])
                else:
                    json_line = json.loads(decoded_line)
            except json.decoder.JSONDecodeError as e:
                print("Error: ", e)
                continue
 
            # properties and items have a similar structure
            element_dict = {}

            # save id only as number for faster querying
            element_dict["_id"] = int(json_line["id"][1:])
            subject_id = element_dict["_id"]
            element_dict["wikidataId"] = json_line["id"]
            # Possibility to use other language codes, default here is "en" for english
            try:
                element_dict["label"] = json_line["labels"]["en"]["value"]
            except KeyError:
                no_english_label_counter += 1
                continue # Don't save items without english label
                #element_dict["label"] = ""
            try: 
                element_dict["description"] = json_line["descriptions"]["en"]["value"]
            except KeyError:
                element_dict["description"] = ""
            try:
                element_dict["aliases"] = [alias["value"] for alias in json_line["aliases"]["en"]]
            except KeyError:
                element_dict["aliases"] = []

            # distinguish between items and properties
            item_type_letter = json_line["id"][0]

            # save claims
            element_dict["properties"] = {}
            for p_id, claim_list in json_line["claims"].items():
                saved_claim = {}
                for i, claim in enumerate(claim_list):
                    mainsnak = claim['mainsnak']
                    # save the first claim
                    if i == 0 and mainsnak["snaktype"] == "value":
                        saved_claim = {'rank': claim["rank"], 'entityType': mainsnak['datatype'], 'dataType': mainsnak['datavalue']['type'], 'value': mainsnak['datavalue']['value']}
                    # save the preferred claim if there are multiple claims
                    if mainsnak["snaktype"] == "value" and claim['rank'] == 'preferred':
                        saved_claim = {'rank': claim["rank"], 'entityType': mainsnak['datatype'], 'dataType': mainsnak['datavalue']['type'], 'value': mainsnak['datavalue']['value']}
                
                    # check if datatype is a wikidata-item
                    if not 'datavalue' in mainsnak.keys():
                        continue
                    if mainsnak['datavalue']['type']=="wikibase-entityid" and mainsnak['datatype']!="wikibase-form" and mainsnak['datatype']!="wikibase-sense":
                        # check if p_id is already in concept_profiling
                        if not p_id in concept_profiling.keys():
                            concept_profiling[p_id] = {}
                        # check if object_id is already in p_id dict
                        object_id = mainsnak['datavalue']['value']["numeric-id"]
                        if not object_id in concept_profiling[p_id].keys():
                            concept_profiling[p_id][object_id] = []
                        # subject is predicate of object
                        concept_profiling[p_id][object_id].append(subject_id)
                
                # saved claim can be empty if the snaktype is 'novalue' or 'somevalue'
                element_dict["properties"][p_id] = saved_claim

            if item_type_letter == "Q" and json_line["type"]=="item":  # is item
                itemsCollection.insert_one(element_dict)

            elif item_type_letter == "P" and json_line["type"]=="property": # is property
                propertiesCollection.insert_one(element_dict)

            num_entities_processed += 1

            if line_number % cfg.checkpoint_iterations == 0:
                log.info(f"Saving checkpoint of concept profiling dict:")
                concept_profiling_path = Path(f"data/profiling/concept_profiling_info_it_{str(line_number)}.json")
                concept_profiling_checkpoints.append(concept_profiling_path)
                with open(concept_profiling_path, "w", encoding="utf8") as file:
                    json.dump(concept_profiling, file, indent=2)
                print(f"Saved checkpoint {line_number}")
 
                # only keep the last checkpoint
                if len(concept_profiling_checkpoints) > 1:
                    os.remove(concept_profiling_checkpoints[0])
                    concept_profiling_checkpoints.pop(0)


    log.info(f"Done, processed {num_entities_processed} wikidata entities")   
    # TODO: add language code that was set in config
    log.info(f"Encountered {no_english_label_counter} items without english label that were therefore skipped")
	   
    # save final profiling dict after finishing preprocessing
    with open(Path(f"data/profiling/concept_profiling_final.json"), "w", encoding="utf8") as file:
        json.dump(concept_profiling, file, indent=2)
    print("Len concept profiling dict:", len(concept_profiling))

