import numpy
import matplotlib
import matplotlib.pyplot as plt


def mcol(v):
    return v.reshape((v.size, 1))


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
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = get_labels(name)
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def load_data():
    D, L = open_file('data/Train.txt')
    return D, L


if __name__ == '__main__':
    load_data()