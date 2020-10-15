import numpy as np
import scipy
from scipy.spatial.distance import euclidean as distance
from functools import lru_cache, cmp_to_key

np.random.seed(0)
points = np.random.random((1000,2))

class KDTree:
    class Node:
        def __init__(self, max_d=2, n=None, d=0):
            self.value = n
            self.left = None
            self.right = None
            self.max_d = max_d
            self.dimension = d // self.max_d

        def insert(self, n):
            d = self.dimension
            if self.value is None:
                self.value = n
            elif n[d] <= self.value[d]:
                if not self.left:
                    self.left = KDTree.Node(self.max_d, n, self.dimension + 1)
                else:
                    self.left.insert(n)
            else:
                if not self.right:
                    self.right = KDTree.Node(self.max_d, n, self.dimension + 1)
                else:
                    self.right.insert(n)

        def print_traversal(self):
            if self.left:
                self.left.print_traversal()
            print(self.value)

            if self.right:
                self.right.print_traversal()

        def find_neighbors(self, p, eps):
            v = self.value
            d = self.dimension
            neighbors = []

            dist = distance(self.value, p)
            if dist > eps:
                if v[d] > p[d]:
                    if self.left is not None:
                        neighbors.extend(self.left.find_neighbors(p, eps))
                else:
                    if self.right is not None:
                        neighbors.extend(self.right.find_neighbors(p, eps))

            else:
            # Recurse into both trees
                neighbors.append(v)
                if self.left is not None:
                    neighbors.extend(self.left.find_neighbors(p, eps))
                if self.right is not None:
                    neighbors.extend(self.right.find_neighbors(p, eps))



            return neighbors




    def __init__(self, dimensions: int, ls=None):
        self.dims = dimensions
        self.head = None
        if ls is not None:
            self.median_insert(ls, 0)

    def insert(self, n):
        if self.head is not None:
            self.head.insert(n)
        else:
            self.head = self.Node(self.dims, n, 0)

    def print_traversal(self):
        if self.head is not None:
            self.head.print_traversal()

    def find_neighbors(self, p, eps):
        if self.head is None:
            return None
        else:
            neighbors = self.head.find_neighbors(p, eps)
            return neighbors

    def median_insert(self, ls, d):
        l = len(ls)

        if l == 1:
            self.insert(ls[0])
            return

        elif l > 1:
            if d == 0:
                key = cmp_to_key(sort_x_key)
            else:
                key = cmp_to_key(sort_y_key)


        if l == 2:
            self.insert(ls[0])
            self.insert(ls[1])
        elif l > 2:
            ls = sorted(ls, key=key)
            m = len(ls) // 2
            self.insert(ls[m])

            self.median_insert(ls[:m], (d + 1) % self.dims)
            self.median_insert(ls[m + 1:], (d + 1) % self.dims)


def sort_x_key(a, b):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]

    if ax == bx:
        return ay - by
    else:
        return ax - bx


def sort_y_key(a, b):

    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]

    if ay == by:
        return ax - bx
    else:
        return ay - by

memo = {}

def find_distance(a, b):
    if a[0] > b[0]:
        a, b = b,a

    global memo

    if (a,b) not in memo:
        memo[(a,b)] = distance(a, b)
    return memo[(a,b)]

if __name__ == "__main__":
    from time import time

    # points = [
    #     (1,1),
    #     (2,2),
    #     (2,1),
    #     (2,3),
    #     (4,5)
    # ]
    tree = KDTree(2, points)


    BF = []
    eps = 1.5
    point = points[0]

    t1 = time()
    NN = tree.find_neighbors(point, eps)
    t1 = time() - t1

    t2 = time()

    for p in points:
        for q in points:
            d = find_distance(tuple(q), tuple(p))
            if d < eps:
                BF.append(p)

    t2 = time() - t2

    print(t1)
    # print(NN)
    print(t2)
    # print(BF)

    print(set([tuple(n) for n in NN]) == set(tuple(n) for n in BF))


