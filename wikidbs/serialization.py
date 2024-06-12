import logging

import cattrs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import torch

log = logging.getLogger(__name__)


converter = cattrs.Converter()

# do not serialize tensors
converter.register_unstructure_hook(torch.Tensor,lambda x: None)  # do not serialize embeddings

# serialize dataframes as JSON
converter.register_unstructure_hook(pd.DataFrame, lambda df: df.to_json())  # serialize dataframe as JSON
converter.register_structure_hook(pd.DataFrame, lambda df_json_str, cls: pd.read_json(df_json_str))  # serialize dataframe as JSON

