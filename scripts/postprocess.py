import ast
import json
import logging
import shutil
import time
from pathlib import Path

import cattrs
import hydra
from omegaconf import DictConfig
from pqdm.processes import pqdm
from tqdm import tqdm

import wikidbs.schema
from wikidbs import utils
from wikidbs.database import Database
from wikidbs.serialization import converter
from wikidbs.visualize import create_schema_diagram

log = logging.getLogger(__name__)


def _load_database(db_path: Path) -> Database:
    with open(db_path / "database.json", "r", encoding="utf-8") as file:
        db = converter.structure(json.load(file), Database)
    for table in db.tables:
        table.table_df.columns = [ast.literal_eval(col) for col in table.table_df.columns]
        if table.llm_renamed_df is not None:
            table.llm_renamed_df.columns = [ast.literal_eval(col) for col in table.llm_renamed_df.columns]
    return db


def postprocess_database(args: tuple[int, Path, Path, DictConfig]) -> None:
    db_idx, db_path, output_path, cfg = args
    db = _load_database(db_path)

    # create directories
    output_db_path = output_path / db_path.name

    output_db_path.mkdir()
    output_tables_path = output_db_path / "tables"
    output_tables_path.mkdir()
    output_tables_with_item_ids_path = output_db_path / "tables_with_item_ids"
    output_tables_with_item_ids_path.mkdir()

    out_tables = []
    for table in db.tables:
        column_types = {col: utils.majority_type(table.table_df[col]) for col in table.table_df.columns}
        out_columns = []
        for col_idx, col_name in enumerate(table.llm_renamed_df.columns):
            col_name = col_name[0]
            orig_column = table.table_df.columns[col_idx]
            col_datatype = column_types[orig_column]
            col_pid = table.table_df.columns[col_idx][1]
            if col_idx <= 1:
                col_pid = None
            out_columns.append(
                wikidbs.schema.Column(
                    column_name=col_name,
                    alt_column_names=[table.llm_only_column_names[col_idx], orig_column[0]],
                    wikidata_property_id=col_pid,
                    data_type=col_datatype
                )
            )

        out_foreign_keys = []
        for fk in table.foreign_keys:
            column_idx = None
            for ix, col in enumerate(table.table_df.columns.to_list()):
                if col[0] == fk.column_name:
                    assert column_idx is None
                    column_idx = ix
            assert column_idx is not None
            column_name = table.llm_renamed_df.columns.to_list()[column_idx][0]  # we want the name, nothing else

            ref_tables = [tab for tab in db.tables if tab.table_name == fk.reference_table_name]
            if len(ref_tables) != 1:
                log.error("FK relationship without reference table! ==> skip")
                continue
            ref_table = ref_tables[0]

            out_foreign_keys.append(
                wikidbs.schema.ForeignKey(
                    column=column_name,
                    reference_column=ref_table.llm_renamed_df.columns.to_list()[0][0],  # we want the name, nothing else
                    reference_table=ref_table.llm_table_name
                )
            )

        out_tables.append(
            wikidbs.schema.Table(
                table_name=table.llm_table_name,
                alt_table_names=[table.llm_only_table_name, table.table_name],
                file_name=f"{table.llm_table_name}.csv",
                columns=out_columns,
                foreign_keys=out_foreign_keys
            )
        )

        filename = f"{table.llm_table_name}.csv"
        table_df_save = table.llm_renamed_df.map(lambda x: x[0] if isinstance(x, list) else x)
        table_df_save.columns = table_df_save.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
        table_df_save.to_csv(output_tables_path / filename, index=False)

        table_df_save = table.llm_renamed_df.map(lambda x: x[1] if isinstance(x, list) else x)
        table_df_save.columns = table_df_save.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
        table_df_save.to_csv(output_tables_with_item_ids_path / filename, index=False)

    out_schema = wikidbs.schema.Schema(
        database_name=db.llm_db_name,
        alt_database_names=[db.llm_only_db_name, db.db_name],
        wikidata_property_id=db.start_table.predicate["id"],
        wikidata_property_label=db.start_table.predicate["label"],
        wikidata_topic_item_id="Q" + db.start_table.object["id"],
        wikidata_topic_item_label=db.start_table.object["label"],
        tables=out_tables
    )

    with open(output_db_path / "schema.json", "w", encoding="utf-8") as file:
        json.dump(cattrs.unstructure(out_schema), file, indent=4)

    for table in out_schema.tables:
        for fk in table.foreign_keys:
            assert fk.column in [c.column_name for c in table.columns]
            ref_tables = [t for t in out_schema.tables if t.table_name == fk.reference_table]
            if len(ref_tables) > 1:
                breakpoint()
            assert len(ref_tables) == 1, len(ref_tables)
            ref_table = ref_tables[0]
            assert fk.reference_column in [c.column_name for c in ref_table.columns]

    create_schema_diagram(tables=out_tables, save_path=output_db_path, show_diagram=False)

    out_schema.validate(output_db_path)


@hydra.main(version_base=None, config_path="../conf", config_name="postprocess.yaml")
def start_postprocessing(cfg: DictConfig):
    # create output folder if necessary
    output_path = Path(f"{cfg.output_folder}")
    if output_path.is_dir():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    dbs_to_process = [x for x in sorted(Path(cfg.input_folder).iterdir()) if x.is_dir()][:cfg.limit]
    log.info(f"Post-process {len(dbs_to_process)} databases from {cfg.input_folder}.")

    args = [(db_idx, db_path, output_path, cfg) for db_idx, db_path in enumerate(dbs_to_process)]

    before = time.time()
    for _ in pqdm(
            args,
            postprocess_database,
            desc="postprocess databases",
            n_jobs=cfg.processes,
            exception_behaviour="immediate"
    ):
        pass
    log.info(f"Done! Processed DBs in {time.time() - before:.2f} seconds.")


if __name__ == "__main__":
    start_postprocessing()
