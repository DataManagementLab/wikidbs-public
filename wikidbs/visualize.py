import seaborn as sns
from pandaserd import ERD
from slugify import slugify  ## python-slugify from https://github.com/un33k/python-slugify
from pathlib import Path
import graphviz
import random

import wikidbs.schema


def visualize_row_statistics(num_rows: list):
    print(f"Have {len(num_rows)} possible topics")

    number_rows_bins = {"1": 0, ">2": 0, ">10": 0, ">100": 0, ">1.000": 0, ">10.000": 0, ">100.000": 0, ">200.000": 0}
    for value in num_rows:
        if value == 1:
            number_rows_bins["1"] += 1
        elif value >= 2 and value < 10:
            number_rows_bins[">2"] += 1
        elif value >= 10 and value < 100:
            number_rows_bins[">10"] += 1
        elif value >= 100 and value < 1000:
            number_rows_bins[">100"] += 1
        elif value >= 1000 and value < 10000:
            number_rows_bins[">1.000"] += 1
        elif value >= 10000 and value < 100000:
            number_rows_bins[">10.000"] += 1
        elif value >= 100000 and value < 200000:
            number_rows_bins[">100.000"] += 1
        elif value >= 200000:
            number_rows_bins[">200.000"] += 1

    sns.set_theme(rc={"figure.figsize":(15, 10)})
    ax = sns.barplot(x=list(number_rows_bins.keys()), y=list(number_rows_bins.values()))
    ax.set(xlabel="Number of rows", ylabel="Number of topics", title="Topics per row number - log scale")
    ax.set_xticklabels(list(number_rows_bins.keys()), rotation=-30)
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.margins(y=0.1)
    ax.set_yscale('log')
    ax.figure.savefig(f'./data/profiling/row_statistics_{len(num_rows)}_topics.png')    
    print(number_rows_bins)


def create_schema_diagram(tables: list[wikidbs.schema.Table], save_path: Path, show_diagram:bool=False):
    """
    Uses the 'pandaserd' library to visualize the schema with all tables in the database and their foreign key relations.

        Args:
            schema_information (dict): A dictionary of all relevant information for the database schema
            save_path (Path): The path to save the resulting diagram in the filesystem
            show_diagram (bool): If the diagram should be opened and shown to the user

        Returns:
            - 
    """
    # visualize schema
    erd_diagram = ERD()
    bg_colors   = ['lightblue', 'skyblue', 'pink', 'lightyellow', 'grey', 'gold']
    # Add tables to ER-diagram 
    for table in tables:
        columns = []
        for col in table.columns:
            columns.append(col.column_name)
        df = pd.DataFrame(columns=columns)
        df = df.reset_index()


        # get foreign keys of table:
        table_fk_columns = [x.column for x in table.foreign_keys]
        table_fk_columns.append(columns[0])

        num_further_columns = len(df.columns) - len(table_fk_columns)

        erd_df = df.filter(table_fk_columns, axis='columns')
        # Add column "x" further columns in table
        erd_df[f"{num_further_columns}_further_columns"] = [0 for x in range(len(erd_df))]

        # adapt names
        erd_df.columns = [slugify(label, word_boundary=True, separator='_', allow_unicode=False, lowercase=False) for label in erd_df.columns]

        table_name = table.table_name

        table_name_slugified = "_" + slugify(table_name, word_boundary=True, separator='_', allow_unicode=False, lowercase=False)
        
        
        erd_diagram.add_table(erd_df, table_name_slugified, bg_color=random.choice(bg_colors))
        
    # add relations:
    for table in tables:
        table_name = table.table_name
        table_name = "_" + slugify(table_name, word_boundary=True, separator='_', allow_unicode=False, lowercase=False)
        
        for fk in table.foreign_keys:
            right_table_name = "_" + slugify(fk.reference_table, word_boundary=True, separator='_', allow_unicode=False, lowercase=False)
            reference_column = slugify(fk.reference_column, word_boundary=True, separator='_', allow_unicode=False, lowercase=False)
            left_table_name = slugify(fk.column, word_boundary=True, separator='_', allow_unicode=False, lowercase=False)
            # right_table_name = fk.reference_table
            #reference_column = fk.reference_column

            erd_diagram.create_rel(left_table_name=table_name, right_table_name=right_table_name, left_on=left_table_name, right_on=reference_column, left_cardinality='1', right_cardinality='1')

    erd_diagram.write_to_file(save_path / f"schema_diagram.dot")

    s = graphviz.Source.from_file(save_path / f"schema_diagram.dot")
    s.render(save_path / "schema_diagram", format="pdf", cleanup=True, view=show_diagram)