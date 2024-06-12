import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

from wikidbs.mongodb import get_db_connection, check_for_existing_collections
from wikidbs.preprocess import preprocess_wikidata_dump


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="preprocess.yaml")
def preprocess(cfg: DictConfig):
    # create data and profiling folder if necessary
    Path("data/profiling").mkdir(parents=True, exist_ok=True)
    
    # make sure that dump exists
    dump_path = Path(cfg.wikidata_dump)
    assert dump_path.exists(), f"Could not find the wikidata dump file at: {dump_path}"

    # get database connection
    db = get_db_connection(cfg, preprocessing=True)

    # initialize collections for items and properties
    # check if the collections already exist, if yes: ask user for confirmation to delete the existing ones
    check_for_existing_collections(db)

    preprocess_wikidata_dump(cfg, db)
    

if __name__=="__main__":
    preprocess()