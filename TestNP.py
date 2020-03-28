import numpy as np
import math
import timeit


filename = "cities50"
cities = np.loadtxt(filename)
n = len(cities)


def method_one():
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] == 0:
                dist[i][j] = euclid(cities[i], cities[j])
                dist[j][i] = euclid(cities[i], cities[j])

    return dist


def method_two():
    dist2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist2[i][j] = euclid(cities[i], cities[j])

    return dist2


def euclid(p, q):
    x = p[0]-q[0]
    y = p[1]-q[1]
    return math.sqrt(x*x+y*y)


def calculate():
    val = 0
    for i in range(n):
        val += dist[perm[i]][perm[(i + 1) % n]]

    print(val)


dist = method_two()
perm = np.arange(0, n)

# Testing for the other way of finding distance
edges = np.loadtxt("sixnodes", dtype=int)

dist3 = np.zeros((6, 6))

for p in range(len(edges)):
    dist3[edges[p][0]][edges[p][1]] = edges[p][2]
    dist3[edges[p][1]][edges[p][0]] = edges[p][2]


# Testing the swap function


def trySwap(i):
    temp = np.copy(perm)
    temp[i], temp[(i+1) % n] = temp[(i+1) % n], temp[i]
    if dist[temp[(i-1) % n]][temp[i]] + dist[temp[(i+1) % n]][temp[(i+2) % n]] < dist[perm[(i-1) % n]][perm[i]] + dist[perm[(i+1) % n]][perm[(i+2) % n]]:
        perm = np.copy(temp)
        return True
    else:
        return False


def swapHeuristic(self):
    better = True
    while better:
        better = False
        for i in range(self.n):
            if self.trySwap(i):
                better = True
