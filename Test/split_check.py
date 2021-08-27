import numpy as np

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)

A = np.array(range(21))
print(A)
A_splitted = split_x(A, 7)
print(A_splitted)