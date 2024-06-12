import attrs

@attrs.define
class ForeignKey:
    column: str
    reference_column: str
    reference_table: str


@attrs.define
class Column:
    column_name: str
    wikidata_property_id: str | None
    data_type: str


@attrs.define
class Table:
    table_name: str
    file_name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey]


@attrs.define
class Schema:
    database_name: str
    wikidata_property_id: str
    wikidata_property_label: str
    wikidata_topic_item_id: str
    wikidata_topic_item_label: str
    tables: list[Table]
