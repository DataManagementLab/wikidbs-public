import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

from wikidbs.convert import convert_profiling_dict_to_jsonlines
from wikidbs.visualize import visualize_row_statistics

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="convert.yaml")
def convert_to_jsonlines(cfg: DictConfig):
    # create data and profiling folder if necessary
    Path("data/profiling").mkdir(parents=True, exist_ok=True)
    
    # make sure that the profiling.json dict exists
    profiling_dict = Path(cfg.profiling_dict)
    assert profiling_dict.exists(), f"Could not find the profiling dict file at: {profiling_dict}"

    log.info(f"Found profiling dict {profiling_dict}, now starting conversion")

    # convert profiling json dict to jsonlines format
    num_rows = convert_profiling_dict_to_jsonlines(cfg)

    # create graph over num_rows distribution    
    visualize_row_statistics(num_rows=num_rows)

if __name__=="__main__":
    convert_to_jsonlines()