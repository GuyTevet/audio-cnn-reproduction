#!/usr/bin/env python

"""
    File name:          dataset_labels.py
    Author:             Guy Tevet
    Date created:       22/12/2017
    Date last modified: 22/12/2017
    Description:        labales handling according csv file
"""

import numpy as np


def get_singers_lookup_table(csv_file):

    lookup = []
    # open csv file
    with open(csv_file, 'rb') as csvfile:

        csvfile.readline() #skip title

        # get number of columns
        for line in csvfile.readlines():
            array = line.split(',')
            lookup.append(int(array[0]))

        #lookup = np.asarray(list(set(lookup)), dtype=np.uint64)

    #with open(output_table , 'w') as file:
    #    np.save(file, lookup)

    return list(set(lookup))