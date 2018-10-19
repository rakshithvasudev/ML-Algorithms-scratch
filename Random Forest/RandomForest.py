# importing the necessary libraries
from random import seed
from random import randrange
from csv import reader
from math import sqrt


def load_csv(filename):
    """
    Loads the csv file into the dataset list
    :param filename: name of the csv file from which content must be read.
    :return: the loaded dataset.
    """
    dataset = []

    with open(filename, 'r') as f:
        csv_reader = reader(f)

        # for each row in the reader, if row exists then add
        # that row to the dataset.
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    return dataset


def str_to_float(dataset, column):
    """
    Converts the dataset entires to float wherever there is an integer.
    :param dataset: the original dataset that needs to be converted as float.
    :param column: the column where the float transformation needs to happen.
    :return:
    """

    for row in dataset:
        if column ==
        row[column] = float(row[column].strip())
