import numpy as np
import pandas as pd
from scipy.stats import stats


def processFile(input):
    # import the CSV
    file = pd.read_csv(input)

    # pre-processing the data, include checking entries, features, checking null values.
    print(input.shape)
    file.head()
    file.isnull().any()
    file.describe()

    # Check to see if data under residual sugar has outlier, because max = 65.8, min = 0.6 and avg = 6.39
    # sns.boxplot(file['residual sugar'])

    # Use Z-Score to remove the outlier and check the remaining entries to be (4487)
    z = np.abs(stats.zscore(file))
    file = file[(z < 3).all(axis=1)]
    # print(input.shape)
    return file
