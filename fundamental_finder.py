import numpy as np

def find_fundamental(x, y):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    index = np.argmax(y)

    return (xmax, ymax, index)
