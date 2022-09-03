import numpy
import matplotlib
import matplotlib.pyplot as plt
from misc.misc import make_column_shape
import misc.constants as constants


def get_labels(name):
    hLabels = {
        '0': 0,
        '1': 1
    }
    return hLabels[name]


def open_file(file_name):
    DList = []
    labelsList = []

    with open(file_name) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = make_column_shape(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = get_labels(name)
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def load_data(filename='Train'):
    D, L = open_file(f'{constants.PWD}/data/{filename}.txt')
    return D, L


if __name__ == '__main__':
    load_data()