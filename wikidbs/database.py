# a database consists of tables that are connected via foreign keys
import collections
import pathlib
import sqlite3
from pathlib import Path

import attrs
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from wikidbs.table import Table, ForeignKey, OutgoingRelation


@attrs.define
class Database:
    tables: list[Table]
    further_relations: list[OutgoingRelation]
    further_relations_from_main_table: list[OutgoingRelation]
    further_relations_from_connected_tables: list[OutgoingRelation]
    foreign_keys: list[ForeignKey]
    semantic_embedding: torch.Tensor | None
    db_name: str | None

    @classmethod
    def from_start_table(cls, start_table: Table) -> "Database":
        return cls(
            tables=[start_table],
            further_relations=start_table.possible_relations.copy(),
            further_relations_from_main_table=start_table.possible_relations.copy(),
            further_relations_from_connected_tables=[],
            foreign_keys=[],
            semantic_embedding=None,
            db_name=None
        )

    @property
    def start_table(self) -> Table:
        return self.tables[0]

    # num_tables # number of tables in the database

    def tables_to_csv(self, path: Path, only_labels: bool=False, use_llm_names: bool=True) -> None:
        for table in self.tables:
            if use_llm_names:
                filename = table.llm_table_name + ".csv"
                if only_labels:
                    table_df_save = table.llm_renamed_df.map(lambda x: x[0] if isinstance(x, list) else x)
                    table_df_save.columns = table_df_save.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
                else:
                    table_df_save = table.llm_renamed_df
            else:
                filename = table.table_name + ".csv"
                if only_labels:
                    table_df_save = table.table_df.map(lambda x: x[0] if isinstance(x, list) else x)
                    table_df_save.columns = table_df_save.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
                else:
                    table_df_save = table.table_df
            table_df_save.to_csv(path / filename, index=False)

    def get_table(self, table_name):
        for table in self.tables:
            if table.table_name == table_name:
                return table

    def get_table_names(self) -> list:
        return [table.table_name for table in self.tables]

    def initialize_semantic_embedding(self, embedding_model: SentenceTransformer):
        self.semantic_embedding = embedding_model.encode(self.start_table.table_name, convert_to_tensor=True,
                                                         show_progress_bar=False)

    def find_semantically_most_similar_relation(self, embedding_model: SentenceTransformer):
        """
        Finds the semantically most similar futher relation to the current database embedding
        """
        # Compute cosine similarity between current database embedding and every relation embedding
        cosine_scores = []
        for relation in self.further_relations:
            cos_sim = util.cos_sim(self.semantic_embedding, relation.semantic_embedding)
            # print(f"Similarity: {relation.property_label} - {cos_sim.item()}")
            cosine_scores.append(cos_sim.item())
            relation.current_similarity = cos_sim.item()

        # get position of maximum score, then take the relation on this position as next relation
        max_idx = cosine_scores.index(max(cosine_scores)) 
        max_score = max(cosine_scores)
        most_similar_relation = self.further_relations[max_idx]
        self.further_relations.pop(max_idx)

        return most_similar_relation, max_score

    def generate_sql(self, val_path: pathlib.Path, ids_path: pathlib.Path, val_db_path, ids_db_path) -> str:
        val_stmts = []
        ids_stmts = []

        with (sqlite3.connect(val_db_path) as val_conn, sqlite3.connect(ids_db_path) as ids_conn):
            val_conn.set_trace_callback(lambda s: val_stmts.append(s))
            ids_conn.set_trace_callback(lambda s: ids_stmts.append(s))

            # change db name
            for table in reversed(self.tables):  # reverse so that the reference constraints work

                ########################################################################################################
                # CREATE TABLE
                ########################################################################################################

                col2datatypes = collections.defaultdict(set)
                for triple in table.llm_renamed_df.columns:
                    for cell_triple in table.llm_renamed_df[triple]:
                        if not pd.isna(cell_triple):
                            col2datatypes[triple[0]].add(cell_triple[2])

                # assert that all values of a column have the same data type
                for llm_col_name, datatypes in col2datatypes.items():
                    assert len(datatypes) == 1

                # assert that no column name ends with "_qid"
                for triple in table.llm_renamed_df.columns:
                    assert not triple[0].endswith("_qid")

                col2datatype = {llm_col_name: list(datatypes)[0] for llm_col_name, datatypes in col2datatypes.items()}

                type_mappings = {  # TODO: improve type mappings
                    "string": "TEXT",
                    "wikibase-entityid": "TEXT",
                    "quantity": "REAL",
                    "globecoordinate": "TEXT",
                    "monolingualtext": "TEXT",
                    "time": "TEXT"
                }

                val_ct_columns, ids_ct_columns = [], []
                fk_constraints = []
                for llm_triple, triple in zip(table.llm_renamed_df.columns, table.table_df.columns):
                    val_ct_columns.append(f""""{llm_triple[0]}" {type_mappings[col2datatype.get(llm_triple[0], "TEXT")]}""")  # TODO: uniqueness/primary key, nullability...
                    ids_ct_columns.append(f""""{llm_triple[0]}_qid" TEXT""")  # QID is always TEXT

                    for foreign_key in self.foreign_keys:
                        if foreign_key.source_table_name == table.table_name and foreign_key.column_name == triple[0]:
                            # determine llm-generated names of the reference table and column
                            llm_ref_table_name = None
                            llm_ref_col_name = None
                            for t in self.tables:
                                if t.table_name == foreign_key.reference_table_name:
                                    assert llm_ref_table_name is None
                                    llm_ref_table_name = t.llm_table_name
                                    for llm_trip, trip in zip(t.llm_renamed_df, t.table_df):
                                        if trip[0] == "label":  # TODO: assumes foreign keys always reference label column
                                            assert llm_ref_col_name is None
                                            llm_ref_col_name = llm_trip[0]

                            fk_constraints.append(f"""FOREIGN KEY ( "{llm_triple[0]}" ) REFERENCES "{llm_ref_table_name}" ( "{llm_ref_col_name}" )""")

                val_conn.execute(f"""CREATE TABLE "{table.llm_table_name}" ({", ".join(val_ct_columns + fk_constraints)});""")
                ids_conn.execute(f"""CREATE TABLE "{table.llm_table_name}" ({", ".join(val_ct_columns + ids_ct_columns + fk_constraints)});""")

                ########################################################################################################
                # INSERT
                ########################################################################################################

                for _, table_row in table.llm_renamed_df.iterrows():
                    val_values = [str(triple[0]) if not pd.isna(triple) else None for triple in table_row]
                    ids_values = [str(triple[1]) if not pd.isna(triple) and not triple[1] is None else None for triple in table_row]  # id is NULL if the value or the id is None

                    val_conn.execute(f"""INSERT INTO "{table.llm_table_name}" VALUES ({", ".join("?" for _ in val_values)});""", val_values)
                    ids_conn.execute(f"""INSERT INTO "{table.llm_table_name}" VALUES ({", ".join("?" for _ in val_values + ids_values)});""", val_values + ids_values)

        val_conn.commit()
        ids_conn.commit()

        with open(val_path, "w", encoding="utf-8") as val_file:
            val_file.write("\n".join(val_stmts))

        with open(ids_path, "w", encoding="utf-8") as ids_file:
            ids_file.write("\n".join(ids_stmts))
