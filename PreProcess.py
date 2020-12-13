import numpy as np
import pandas as pd
from scipy.stats import stats

pd.set_option('display.max_columns', None)


def processFile(input):
    # import the CSV
    file = pd.read_csv(input)

    # pre-processing the data, include checking entries, features, checking null values.
    file.head()
    file.isnull().any()
    print(file.isnull().any())
    print(file.describe())
    # Use Z-Score to remove the outlier and check the remaining entries to be (4487)
    z = np.abs(stats.zscore(file))
    file = file[(z < 3).all(axis=1)]
    # print(input.shape)
    return file


