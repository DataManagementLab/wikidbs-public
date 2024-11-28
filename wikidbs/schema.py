import json
import pathlib

import attrs
import cattrs
import pandas as pd


def _validate_identifier(identifier: str) -> None:
    assert isinstance(identifier, str)
    assert identifier != ""


def _validate_table(table: pd.DataFrame) -> None:
    assert len(table.columns) > 0
    assert len(table.index) > 0


@attrs.define
class ForeignKey:
    column: str
    reference_column: str
    reference_table: str

    def validate(self) -> None:
        _validate_identifier(self.column)
        _validate_identifier(self.reference_column)
        _validate_identifier(self.reference_table)


@attrs.define
class Column:
    column_name: str
    alt_column_names: list[str]  # length 2, not unique
    wikidata_property_id: str | None
    data_type: str

    def validate(self) -> None:
        _validate_identifier(self.column_name)

        assert isinstance(self.alt_column_names, list) and len(self.alt_column_names) == 2
        for alt_column_name in self.alt_column_names:
            _validate_identifier(alt_column_name)

        if self.wikidata_property_id is not None:
            _validate_identifier(self.wikidata_property_id)

        _validate_identifier(self.data_type)


@attrs.define
class Table:
    table_name: str
    alt_table_names: list[str]  # length 2, not unique
    file_name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey]

    def validate(self, db_path: pathlib.Path) -> None:
        _validate_identifier(self.table_name)

        assert isinstance(self.alt_table_names, list) and len(self.alt_table_names) == 2
        for alt_table_name in self.alt_table_names:
            _validate_identifier(alt_table_name)

        assert isinstance(self.file_name, str) and len(self.file_name) > 4 and self.file_name.endswith(".csv")

        assert isinstance(self.columns, list) and len(self.columns) > 0
        column_names_lower = set()
        for column in self.columns:
            assert isinstance(column, Column)
            column.validate()
            assert column.column_name.lower() not in column_names_lower
            column_names_lower.add(column.column_name.lower())

        assert isinstance(self.foreign_keys, list)
        column_names = {column.column_name for column in self.columns}
        for foreign_key in self.foreign_keys:
            assert isinstance(foreign_key, ForeignKey)
            foreign_key.validate()
            assert foreign_key.column in column_names

        table_path = db_path / "tables" / self.file_name
        assert table_path.is_file()
        df = pd.read_csv(table_path)
        _validate_table(df)
        assert df.columns.to_list() == [column.column_name for column in self.columns]

        table_with_item_ids_path = db_path / "tables_with_item_ids" / self.file_name
        assert table_with_item_ids_path.is_file()
        df_ids = pd.read_csv(table_with_item_ids_path)
        _validate_table(df_ids)
        assert df_ids.columns.to_list() == [column.column_name for column in self.columns]

        assert df.index.equals(df_ids.index)
        assert df.columns.equals(df_ids.columns)


@attrs.define
class Schema:
    database_name: str  # not unique, only later
    alt_database_names: list[str]  # length 2, not unique
    wikidata_property_id: str
    wikidata_property_label: str
    wikidata_topic_item_id: str
    wikidata_topic_item_label: str  # is sometimes an empty string
    tables: list[Table]

    def validate(self, db_path: pathlib.Path) -> None:
        _validate_identifier(self.database_name)

        assert isinstance(self.alt_database_names, list) and len(self.alt_database_names) == 2
        for alt_database_name in self.alt_database_names:
            _validate_identifier(alt_database_name)

        _validate_identifier(self.wikidata_property_id)
        _validate_identifier(self.wikidata_property_label)
        _validate_identifier(self.wikidata_topic_item_id)
        assert isinstance(self.wikidata_topic_item_label, str)  # is sometimes empty string

        assert isinstance(self.tables, list) and len(self.tables) > 0
        table_names_lower = set()
        for table in self.tables:
            assert isinstance(table, Table)
            table.validate(db_path)
            assert table.table_name.lower() not in table_names_lower
            table_names_lower.add(table.table_name.lower())

        table_names = {table.table_name for table in self.tables}
        for table in self.tables:
            for foreign_key in table.foreign_keys:
                assert foreign_key.reference_table in table_names
                ref_table = [tab for tab in self.tables if tab.table_name == foreign_key.reference_table][0]
                assert foreign_key.reference_column in {column.column_name for column in ref_table.columns}

        schema_path = db_path / "schema.json"
        assert schema_path.is_file()
        with open(schema_path, "r", encoding="utf-8") as file:
            _ = cattrs.structure(json.load(file), Schema)

        schema_diagram_pdf_path = db_path / "schema_diagram.pdf"
        assert schema_diagram_pdf_path.is_file()
        schema_diagram_dot_path = db_path / "schema_diagram.dot"
        assert schema_diagram_dot_path.is_file()
