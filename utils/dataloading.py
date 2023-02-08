import pandas as pd
import numpy as np
from typing import List, Tuple

def read_kinetic(path: str, features: List[str], targets: List[str], sort_by: str = 'relTime') -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorts the kinetic .csv file by 'relTime' and returns needed columns
    
    Args:
        path: path to the .csv kinetic file
        features: list of features to be in the X part of the return
        targets: list of features to be in the y part of the return
    Returns:
        X, y: features and targets in the numpy format
    """
    df = pd.read_csv(path).sort_values(sort_by)
    return df[features].values, df[targets].values