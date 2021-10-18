'''
tt
2021-10-14
Decision tree: ID.3 algorithm
'''

import numpy as np
import csv
import math

def read_data(path="./weather.csv"):
    data = {}
    fields = []
    with open(path) as f:
        csv_reader = csv.reader(f)
        is_header = True
        for line in csv_reader:
            if is_header:
                for field in line:
                    fields.append(field)
                    data[field] = []
                is_header = False
            else:
                for i in range(len(line)):
                    data[fields[i]].append(line[i])
    return fields, data

def get_entropy(data, category_field, field, field_value):
    '''
    Compute entropy value of a specific value for a specific field
    :param data: dataset
    :param category_field: which field is category label
    :param field: the specific field
    :param field_value: the specific value
    :return:
    '''

    # 1. First collect categories for all those whose field == field_value
    labels = {}
    for i in range(len(data[category_field])):
        if data[field] == field_value:
            labels[data[category_field][i]] += 1


    # 2. Compute entropy
    entropy = 0.
    for label, count in labels.items():
        pi = (count) / sum(labels.values())
        entropy += -1 * pi * math.log(pi, 2)

    assert entropy <= 1
    return entropy


def id3(fields, data):
    pass


if __name__ == "__main__":
    fields, data = read_data()
    
