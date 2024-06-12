from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

import logging

log = logging.getLogger(__name__)

def get_db_connection(cfg, preprocessing: bool=False):
    """
        Get the MongoClient for the given database.
        
        Raises an error if the database doesn't exist during non-preprocessing phases.
    """
    log.info("Trying to connect to database")
    # get database client and connect to database
    client = init_db_client(cfg)

    # get database name and check if a database with that name already exists
    db_name = cfg.db.database
    if db_name in client.list_database_names():
        log.info(f"Found the *{db_name}* database.")
    else:
        if preprocessing:
            log.info(f"No DB yet with name *{db_name}* yet.")
        else:
            log.info(f"No DB yet with name *{db_name}*, please run the preprocessing first!")
            raise FileNotFoundError

    return client[db_name]


def init_db_client(cfg):
    """ 
        Initialize database connection 
    """
    client = MongoClient(**cfg.db.config)

    try:
        ping_result = client.admin.command("ping")
        if ping_result["ok"]:
            log.info("Database connection established!")

    except ConnectionFailure as error:
        log.error("Error: Couldn't connect to database")
        raise Warning(error)

    return client


def check_for_existing_collections(db):
    collections = ["items", "properties"]

    # check if the collections already exist, if yes: ask user for confirmation to delete the existing ones
    userInput = None
    for collection in collections:
        if collection in db.list_collection_names():
            while userInput is None or (userInput.lower().strip() not in ["y","yes","ja","n","no","nope"]):
                userInput = input(f"WARNING: there are collections for this preprocessing already in the database. To run the preprocessing again, the previous collections will be deleted. Proceed? (Y/N) \n")
                if userInput.lower() in ["n","no","nope"]:
                    log.info("Aborting preprocessing")
                    raise NotImplementedError
            if userInput.lower() in ["y","yes","ja"]:
                collectionObj = db[collection]
                log.info(f"Info: Dropping collection {collection}")
                collectionObj.drop()