import collections
import json
import logging
import pathlib
from typing import Any

import attrs
import cattrs
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from pqdm.processes import pqdm

from wikidbs.wikidbs import Schema

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    path: pathlib.Path = "add-path-to-dbs" # Please add your path here!
    processes: int = 64
    limit: int | None = None


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    logger.info(f"start gathering statistics")

    all_db_paths = list(sorted(cfg.path.glob("*/*/")))
    logger.info(f"globbed {len(all_db_paths)} database directories")

    # iterate through databases
    all_stats = {
        "num_tables": [],
        "num_cols": [],
        "num_rows": [],
        "sparsities": [],
        "num_numerical": [],
        "num_non_numerical": [],
        "tab_names": collections.Counter(),
        "col_names": collections.Counter(),
        "alt_0_tab_names": collections.Counter(),
        "alt_1_tab_names": collections.Counter(),
        "alt_0_col_names": collections.Counter(),
        "alt_1_col_names": collections.Counter()
    }
    for stats in pqdm(all_db_paths, handle_db_path, n_jobs=cfg.processes, exception_behaviour="immediate"):
        if stats is not None:
            for key, values in stats.items():
                all_stats[key] += values

    # save statistics
    for key in ("tab_names", "col_names", "alt_0_tab_names", "alt_1_tab_names", "alt_0_col_names", "alt_1_col_names"):
        all_stats[key] = dict(all_stats[key])

    with open(f"statistics.json", "w", encoding="utf-8") as file:
        json.dump(all_stats, file)

    logger.info("done!")


def handle_db_path(db_path: pathlib.Path) -> dict[str, Any]:
    stats = {
        "num_tables": [],
        "num_cols": [],
        "num_rows": [],
        "sparsities": [],
        "num_numerical": [],
        "num_non_numerical": [],
        "tab_names": collections.Counter(),
        "col_names": collections.Counter(),
        "alt_0_tab_names": collections.Counter(),
        "alt_1_tab_names": collections.Counter(),
        "alt_0_col_names": collections.Counter(),
        "alt_1_col_names": collections.Counter()
    }

    with open(db_path / "schema.json", "r", encoding="utf-8") as file:
        schema = cattrs.structure(json.load(file), Schema)

    stats["num_tables"].append(len(schema.tables))
    for table in schema.tables:
        stats["num_cols"].append(len(table.columns))
        df = pd.read_csv(db_path / "tables" / table.file_name)
        stats["num_rows"].append(len(df.index))
        num_numerical = 0
        num_non_numerical = 0
        for dtype in df.dtypes.to_list():
            if dtype.kind in ("i", "f", "u"):
                num_numerical += 1
            else:
                num_non_numerical += 1
        stats["num_numerical"].append(num_numerical)
        stats["num_non_numerical"].append(num_non_numerical)
        stats["sparsities"].append(compute_table_sparsity(df))

        stats["tab_names"][table.table_name] += 1
        stats["alt_0_tab_names"][table.alternative_table_names[0]] += 1
        stats["alt_1_tab_names"][table.alternative_table_names[1]] += 1
        for column in table.columns:
            stats["col_names"][column.column_name] += 1
            stats["alt_0_col_names"][column.alternative_column_names[0]] += 1
            stats["alt_1_col_names"][column.alternative_column_names[1]] += 1

    return stats


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
