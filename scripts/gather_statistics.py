import collections
import json
import logging
import pathlib
import time
from typing import Any

import attrs
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from pqdm.processes import pqdm

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    path: pathlib.Path = "/home/mbodensohn/Source/wikidbs/wikidbs_cration/final_dbs_for_training"
    n_jobs: int = 16
    limit: int | None = None


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    # glob database directories
    before = time.time()
    all_db_paths = list(sorted(cfg.path.glob("*/")))[:cfg.limit]
    logger.info(f"Globbed {len(all_db_paths)} database directories in {time.time() - before:.3f} seconds.")

    # iterate through databases
    before = time.time()
    successes = collections.Counter()
    all_stats = {
        "num_tables": [],
        "num_cols": [],
        "num_rows": [],
        "sparsities": [],
        "num_numerical": [],
        "num_non_numerical": [],
        "tab_names": collections.Counter(),
        "col_names": collections.Counter(),
        "llm_tab_names": collections.Counter(),
        "llm_col_names": collections.Counter()
    }
    for stats in pqdm(all_db_paths, handle_db_path, n_jobs=10, exception_behaviour="immediate"):
        successes[stats is not None] += 1
        if stats is not None:
            for key, values in stats.items():
                all_stats[key] += values
    logger.info(f"Gathering stats successful: {successes}")
    logger.info(f"Iterated databases in {time.time() - before:.3f} seconds.")

    # save statistics
    before = time.time()
    for key in ("tab_names", "col_names", "llm_tab_names", "llm_col_names"):
        all_stats[key] = dict(all_stats[key])

    with open(f"data/stats.json", "w", encoding="utf-8") as file:
        json.dump(all_stats, file)
    logger.info(f"Saved statistics in {time.time() - before:.3f} seconds.")

    logger.info("Done!")


def handle_db_path(db_path: pathlib.Path) -> dict[str, Any] | None:
    try:
        stats = {
            "num_tables": [],
            "num_cols": [],
            "num_rows": [],
            "sparsities": [],
            "num_numerical": [],
            "num_non_numerical": [],
            "tab_names": collections.Counter(),
            "col_names": collections.Counter(),
            "llm_tab_names": collections.Counter(),
            "llm_col_names": collections.Counter()
        }

        table_paths = list(sorted(db_path.joinpath("tables_wikidata").glob("*.csv")))
        stats["num_tables"].append(len(table_paths))
        for table_path in table_paths:
            df = pd.read_csv(table_path)
            stats["num_cols"].append(len(df.columns))
            stats["num_rows"].append(len(df.index))

            stats["tab_names"][table_path.name[:-4]] += 1
            for col_name in df.columns.to_list():
                stats["col_names"][col_name] += 1

            stats["sparsities"].append(compute_table_sparsity(df))

            num_numerical = 0
            num_non_numerical = 0
            for dtype in df.dtypes.to_list():
                if dtype.kind in ("i", "f", "u"):
                    num_numerical += 1
                else:
                    num_non_numerical += 1
            stats["num_numerical"].append(num_numerical)
            stats["num_non_numerical"].append(num_non_numerical)

        llm_table_dir_path = db_path.joinpath("tables_llm")
        if not llm_table_dir_path.exists():
            llm_table_dir_path = db_path.joinpath("tables_wikidata")
        llm_table_paths = llm_table_dir_path.glob("*.csv")
        for llm_table_path in llm_table_paths:
            llm_df = pd.read_csv(llm_table_path)

            stats["llm_tab_names"][llm_table_path.name[:-4]] += 1
            for llm_col_name in llm_df.columns.to_list():
                stats["llm_col_names"][llm_col_name] += 1

        return stats

    except:
        logger.warning(f"Failed to gather stats for {db_path}.", exc_info=True)

    return None


def compute_table_sparsity(df: pd.DataFrame) -> float:
    """Compute the sparsity of the given table as the fraction of values that is nan.

    Args:
        df: The given table.

    Returns:
        The table sparsity.
    """
    return df.isna().sum().sum() / (len(df.index) * len(df.columns))


if __name__ == "__main__":
    main()
