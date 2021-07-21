import numpy as np

a = np.array(range(1, 11))
size_a = 5

# A function that returns metrics which has timestep(=size - 1, feature) on the left and designated y on the right
def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)    

dataset = split_x(a, size_a)

# x = dataset[:, : size - 1], y = dataset[:, size - 1] 
x = dataset[:, :4]
y = dataset[:, 4]

'''
Dataset:
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]
 x:             y:
  [[1 2 3 4]     [5
   [2 3 4 5]      6
   [3 4 5 6]      7
   [4 5 6 7]      8
   [5 6 7 8]      9
   [6 7 8 9]]     10]
'''