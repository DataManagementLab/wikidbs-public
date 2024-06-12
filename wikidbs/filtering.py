import logging

log = logging.getLogger(__name__)


def prune_sparse_columns(dataframe, null_threshold: float):
    """
    Prunes the dataframe in the following ways:
        - removes too sparse columns where more than null_threshold percent values are missing

        Args:
            dataframe (Dataframe): The dataframe to be filtered
            null_threshold (float): Percentage of null values in a column, columns with a higher percentage get deleted

        Returns:
            dataframe: The filtered dataframe
    """
    min_num_values = int(len(dataframe) * (1 - null_threshold))
    log.debug(f"Columns must have at least {min_num_values} values to be kept")
    dataframe = dataframe.dropna(axis="columns", thresh=min_num_values)
    return dataframe


def filter_id_columns(dataframe):
    """
    Removes columns that contain the word 'ID' from the dataframe

        Args: 
            dataframe (Dataframe): The dataframe to be filtered

        Returns:
            dataframe: The filtered dataframe
    """
    dataframe = dataframe[dataframe.columns[~dataframe.columns.str.contains("ID")]]
    return dataframe