import itertools
import json
import logging
import os
import pathlib
import random
import shutil

import attrs
import cattrs
import hydra
import pandas as pd
import tqdm.asyncio
from hydra.core.config_store import ConfigStore
from pqdm.processes import pqdm

import wikidbs.wikidbs
import wikidbs.schema
from wikidbs.visualize import create_schema_diagram

logger = logging.getLogger(__name__)

#####################################################################
#  Script to finalize the dataset in the WikiDBs format
#####################################################################


@attrs.define
class Config:
    renamed_dir: pathlib.Path = "add-path" # Add path to the renamed DBs here
    finalized_dir: pathlib.Path = "add-path" # Add path where to save the finalized DBs
    processes: int = 64
    total_size: int = 100000 # number of DBs
    part_size: int = 20000 # number of DBs per split


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    logger.info(f"start finalizing databases")

    all_renamed_paths = list(sorted(cfg.renamed_dir.glob("*/")))
    logger.info(f"globbed {len(all_renamed_paths)} database directories")

    if cfg.finalized_dir.is_dir():
        logger.info("remove existing directory for finalized databases")
        shutil.rmtree(cfg.finalized_dir)
    cfg.finalized_dir.mkdir(parents=True)

    logger.info("determine database parameters")
    db_parameters = []
    for db_params in pqdm(all_renamed_paths, gather_db_params, n_jobs=cfg.processes, exception_behaviour="immediate"):
        db_parameters.append(db_params)

    db_parameters.sort()  # sorted by sparsity
    db_parameters = db_parameters[:cfg.total_size]

    random.seed(364877888)
    random.shuffle(db_parameters)
    logger.info("shuffle databases")

    part_idx, part_size = 0, 0
    part_dir = cfg.finalized_dir / f"part-{part_idx}"
    part_dir.mkdir()
    lower_db_names = set()
    num_renamed_databases = 0
    for db_idx, (sparsity, db_path) in enumerate(tqdm.tqdm(db_parameters, desc="process databases")):
        with open(db_path / "schema.json", "r", encoding="utf-8") as file:
            schema = cattrs.structure(json.load(file), wikidbs.schema.Schema)

        if schema.database_name.lower() in lower_db_names:
            num_renamed_databases += 1
            mode = ""
            if "_" in schema.database_name:
                mode = "_"
            elif any("_" in table.table_name for table in schema.tables):
                mode = "_"
            elif any("_" in column.column_name for table in schema.tables for column in table.columns):
                mode = "_"
            for idx in itertools.count(start=1):
                name = f"{schema.database_name}{mode}{idx}"
                if name.lower() not in lower_db_names:
                    schema.database_name = name
                    break
        lower_db_names.add(schema.database_name.lower())

        output_db_path = part_dir / f"{str(db_idx).zfill(5)} {schema.database_name}"
        shutil.copytree(db_path, output_db_path)

        create_schema_diagram(schema.tables, output_db_path)
        shutil.move(output_db_path / "schema_diagram.pdf", output_db_path / "schema.pdf")
        os.remove(output_db_path / "schema_diagram.dot")
        os.remove(output_db_path / "database.db")
        os.remove(output_db_path / "schema.json")

        out_tables = []
        for table in schema.tables:
            out_columns = []
            for column in table.columns:
                out_columns.append(wikidbs.wikidbs.Column(
                    column_name=column.column_name,
                    alternative_column_names=column.alt_column_names,
                    data_type=column.data_type,
                    wikidata_property_id=column.wikidata_property_id
                ))

            out_foreign_keys = []
            for fk in table.foreign_keys:
                out_foreign_keys.append(wikidbs.wikidbs.ForeignKey(
                    column_name=fk.column,
                    reference_column_name=fk.reference_column,
                    reference_table_name=fk.reference_table
                ))

            out_tables.append(
                wikidbs.wikidbs.Table(
                    table_name=table.table_name,
                    alternative_table_names=table.alt_table_names,
                    file_name=table.file_name,
                    columns=out_columns,
                    foreign_keys=out_foreign_keys
                )
            )

        out_schema = wikidbs.wikidbs.Schema(
            database_name=schema.database_name,
            alternative_database_names=schema.alt_database_names,
            wikidata_property_id=schema.wikidata_property_id,
            wikidata_property_label=schema.wikidata_property_label,
            wikidata_topic_item_id=schema.wikidata_topic_item_id,
            wikidata_topic_item_label=schema.wikidata_topic_item_label,
            tables=out_tables
        )

        with open(output_db_path / "schema.json", "w", encoding="utf-8") as file:
            json.dump(cattrs.unstructure(out_schema), file, indent=2)

        part_size += 1
        if part_size == cfg.part_size:
            part_size = 0
            part_idx += 1
            part_dir = cfg.finalized_dir / f"part-{part_idx}"
            part_dir.mkdir()

    logger.info(f"had to rename {num_renamed_databases} for uniqueness")


def gather_db_params(db_path: pathlib.Path) -> tuple[str, float, pathlib.Path]:
    with open(db_path / "schema.json", "r", encoding="utf-8") as file:
        schema = cattrs.structure(json.load(file), wikidbs.schema.Schema)

    schema.validate(db_path)

    num_cells = 0
    num_empty_cells = 0
    for table in schema.tables:
        df = pd.read_csv(db_path / "tables" / table.file_name)
        num_cells += (len(df.index) * len(df.columns))
        num_empty_cells += df.isna().sum().sum()
    sparsity = num_empty_cells / num_cells  # db-level sparsity

    return sparsity, db_path


if __name__ == "__main__":
    main()
