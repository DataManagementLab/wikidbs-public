import glob
import json
import os
import sqlite3

import attrs
import cattrs
import pandas as pd
import tqdm


@attrs.define
class ForeignKey:
    column_name: str
    reference_column_name: str
    reference_table_name: str


@attrs.define
class Column:
    column_name: str  # unique in table
    alternative_column_names: list[str]  # length 2, not unique
    data_type: str
    wikidata_property_id: str | None


@attrs.define
class Table:
    table_name: str  # unique in database
    alternative_table_names: list[str]  # length 2, not unique
    file_name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey]


@attrs.define
class Schema:
    database_name: str  # unique across databases
    alternative_database_names: list[str]  # length 2, not unique
    wikidata_property_id: str
    wikidata_property_label: str
    wikidata_topic_item_id: str
    wikidata_topic_item_label: str
    tables: list[Table]


########################################################################################################################
# script to create sqlite databases
########################################################################################################################

if __name__ == "__main__":
    db_paths = list(sorted(glob.glob("*/*/")))
    print(f"create sqlite databases for {len(db_paths)} databases")

    for db_path in tqdm.tqdm(db_paths, desc="create sqlite databases"):
        with open(f"{db_path}/schema.json", "r", encoding="utf-8") as file:
            schema = cattrs.structure(json.load(file), Schema)

        sqlite_path = f"{db_path}/database.db"
        if os.path.isfile(sqlite_path):
            os.remove(sqlite_path)

        with sqlite3.connect(sqlite_path) as conn:
            for table in schema.tables:
                # create table
                ct_columns = []
                for column in table.columns:
                    ct_columns.append(f"\"{column.column_name}\"")

                fk_constraints = []
                for fk in table.foreign_keys:
                    fk_constraints.append(
                        f"FOREIGN KEY ( \"{fk.column_name}\" ) "
                        f"REFERENCES \"{fk.reference_table_name}\" ( \"{fk.reference_column_name}\" )"
                    )

                conn.execute(f"CREATE TABLE \"{table.table_name}\" ( {', '.join(ct_columns + fk_constraints)} );")
                conn.commit()

                # insert data
                df = pd.read_csv(f"{db_path}/tables/{table.file_name}")
                try:
                    df.to_sql(table.table_name, conn, if_exists="append", index=False)
                except OverflowError:  # some integers are too large for sqlite
                    conn.execute(f"DELETE FROM \"{table.table_name}\";")
                    conn.commit()
                    df = df.map(str)
                    df.to_sql(table.table_name, conn, if_exists="append", index=False)

            conn.commit()

    print("done!")
