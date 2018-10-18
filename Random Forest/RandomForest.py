# importing the necessary libraries
from random import seed
from random import randrange
from csv import reader
from math import sqrt


def load_csv(filename):

    dataset = []

    with open(filename,'r') as f:
        csv_reader = reader(f)

        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    return dataset

def str_to_float(dataset, column):

    for row in dataset:

        row[column] = float(row[column].strip())
