import pandas as pd
import attrs
import torch

from sentence_transformers import SentenceTransformer

from wikidbs.filtering import prune_sparse_columns


@attrs.define
class ForeignKey:
    source_table_name: str
    property_id: str
    column_name: str
    reference_table_name: str
    similarity: float


@attrs.define
class OutgoingRelation:
    parent_table: str
    property_id: str
    property_label: str
    semantic_embedding: torch.Tensor | None
    distinct_entities: pd.DataFrame
    current_similarity: float


@attrs.define
class Table:
    """
    A single table from the dataset.

    """
    table_name: str
    predicate: dict  # dict[str, str | int | None]
    object: dict  # dict[str, str | int | None]
    row_entity_ids: list
    rows: list | None
    full_properties_with_outgoing_items: set | None
    properties_with_outgoing_items: set | None
    foreign_keys: list[ForeignKey]
    llm_table_name: str | None
    llm_renamed_df: pd.DataFrame | None
    table_df: pd.DataFrame | None
    possible_relations: list[OutgoingRelation] | None
    llm_only_table_name: str | None = None
    llm_only_column_names: list[str] | None = None

    @classmethod
    def create(cls,
               table_name: str,
               row_entity_ids: list,
               rows: list,
               properties_with_outgoing_items: set,
               predicate_id: int=None,
               predicate: str=None,
               object: str=None,
               object_id: int=None):
        """
        Initialize the table.

        Tables are built by the following relation:  [subjects] are predicate of object
        """
        return cls(
            table_name=table_name,
            predicate={"id": predicate_id, "label": predicate},
            object={"id": object_id, "label": object},
            row_entity_ids=row_entity_ids,
            rows=rows,
            full_properties_with_outgoing_items=properties_with_outgoing_items,
            properties_with_outgoing_items=None,
            foreign_keys=[],
            llm_table_name=None,
            llm_renamed_df=None,
            table_df=None,
            possible_relations=None
        )

    @property
    def columns(self) -> pd.Index:
        return self.table_df.columns

    def transform_to_dataframe(self, sparsity: float) -> None:
        """
        Prune sparse columns and transform the rows list to a pandas DataFrame.

        Every column and cell of the DataFrame still contains the triple format (label, q_id, datatype)
        """
        dataframe_data =  [{key: value for key, value in current_row.items()} for current_row in self.rows]
        full_table_df = pd.DataFrame(dataframe_data)

        self.table_df = prune_sparse_columns(full_table_df, sparsity)
        self.properties_with_outgoing_items = set([x for x in self.full_properties_with_outgoing_items if x in self.table_df.columns])


    def find_outgoing_relations(self, min_distinct_values_in_column: int, embedding_model: SentenceTransformer):
        self.possible_relations = []
        for (property_label, property_id, _) in self.properties_with_outgoing_items:
            column_distinct_items = self.table_df[(property_label, property_id, "datatype") ].dropna().drop_duplicates()
            if len(column_distinct_items) >= int(min_distinct_values_in_column):
                relation_embedding = embedding_model.encode(property_label, convert_to_tensor=True, show_progress_bar=False)
                relation = OutgoingRelation(parent_table=self.table_name,
                                            property_id=property_id,
                                            property_label=property_label,
                                            semantic_embedding=relation_embedding,
                                            distinct_entities=column_distinct_items,
                                            current_similarity=0)
                self.possible_relations.append(relation)
            else:
                #print(f"Got only {len(column_distinct_items)} for {property_label}")
                pass
