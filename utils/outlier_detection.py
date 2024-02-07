import numpy as np


def remove_outliers_3sigma(data):

    """remove_outliers_3sigma

    :param data: data to remove outliers
    :return cleaned_data: cleaned data
    :return removed_data: removed data
    """

    # mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # outlier threshold
    threshold = 3 * std
    removed_data = []
    cleaned_data = []

    for value in data:
        if abs(value - mean) <= threshold:
            cleaned_data.append(value)
        else:
            removed_data.append(value)

    return cleaned_data, removed_data
