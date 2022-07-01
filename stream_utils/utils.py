import os
import re
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

DS_FOLDER = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "datasets")


def list_datasets() -> List[str]:
    """
    List CSV datasets in the dataset folder.

    :return: list of CSV file names
    :rtype: List[str]
    """
    return [f for f in os.listdir(DS_FOLDER) if f.endswith(".csv")]


def read_data(dataset: str) -> pd.DataFrame:
    """
    Reads any desired dataset

    :param dataset: name of the file to read
    :type dataset: str
    :return: dataframe which have been read
    :rtype: pd.DataFrame
    """
    path = os.path.join(DS_FOLDER, dataset)
    return pd.read_csv(path)

def get_features(dataset: pd.DataFrame) -> List[str] :

    return dataset.columns