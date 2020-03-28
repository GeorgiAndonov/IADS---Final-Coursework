import math
import numpy as np


def euclid(p, q):
    x = p[0] - q[0]
    y = p[1] - q[1]
    return math.sqrt(x * x + y * y)


class Graph:

    # Complete as described in the specification, taking care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self, n, filename):
        if n == -1:
            cities = np.loadtxt(filename)
            self.n = len(cities)
            self.perm = np.arange(0, self.n)
            self.dist = np.zeros((self.n, self.n))
            # for i in range(n):
            #     for j in range(n):
            #         if i != j and self.dist[i][j] == 0:
            #             self.dist[i][j] = euclid(cities[i], cities[j])
            #             self.dist[j][i] = euclid(cities[i], cities[j])
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.dist[i][j] = euclid(cities[i], cities[j])
        else:
            self.n = n
            edges = np.loadtxt(filename, dtype=int)
            self.perm = np.arange(0, self.n)
            self.dist = np.zeros((self.n, self.n))
            for p in range(len(edges)):
                self.dist[edges[p][0]][edges[p][1]] = edges[p][2]
                self.dist[edges[p][1]][edges[p][0]] = edges[p][2]

    # Complete as described in the spec, to calculate the cost of the
    # current tour (as represented by self.perm).
    def tourValue(self):
        val = 0
        for i in range(len(self.perm)):
            val += self.dist[self.perm[i]][self.perm[(i + 1) % self.n]]
        return val

    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self, i):
        temp = np.copy(self.perm)
        temp[i], temp[(i + 1) % self.n] = temp[(i + 1) % self.n], temp[i]
        if self.dist[temp[(i - 1) % self.n]][temp[i]] + \
                self.dist[temp[(i + 1) % self.n]][temp[(i + 2) % self.n]] < \
                self.dist[self.perm[(i - 1) % self.n]][self.perm[i]] + \
                self.dist[self.perm[(i + 1) % self.n]][self.perm[(i + 2) % self.n]]:
            self.perm = np.copy(temp)
            return True
        else:
            return False

    # Consider the effect of reversiing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.              
    def tryReverse(self,i,j): # Use temp again with split and the reverse properties of numpy arrays
        temp = np.copy(self.perm)
        temp = np.concatenate((temp[:i], np.flip(temp[i:j+1]), temp[j+1:]))
        if self.dist[temp[(i - 1) % self.n]][temp[i]] + \
                self.dist[temp[j]][temp[(j + 1) % self.n]] < \
                self.dist[self.perm[(i - 1) % self.n]][self.perm[i]] + \
                self.dist[self.perm[j]][self.perm[(j + 1) % self.n]]:
            self.perm = np.copy(temp)
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

    def TwoOptHeuristic(self):
        better = True
        while better:
            better = False
            for j in range(self.n - 1):
                for i in range(j):
                    if self.tryReverse(i, j):
                        better = True

    # Implement the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.
    def Greedy(self):
        perm = [0]
        curr_item = 0
        temp = list(range(1, self.n))
        iter_l = 0
        while iter_l < self.n - 1:
            min_dist = self.dist[curr_item][temp[0]]
            next_item = temp[0]
            for t in temp:
                if min_dist > self.dist[curr_item][t]:
                    min_dist = self.dist[curr_item][t]
                    next_item = t
            perm.append(next_item)
            temp.remove(next_item)
            curr_item = next_item
            iter_l += 1

        self.perm = np.copy(np.array(perm))

    def nearest_insertion(self):
        # Find the smallest distance between 2 cities and the 2 cities
        perm = []
        temp = list(range(0, self.n))
        min_dist = self.dist[0][1]
        min_dist_city1 = 0
        min_dist_city2 = 0
        for ix, iy in np.ndindex(self.dist.shape):
            if ix != iy and min_dist > self.dist[ix][iy]:
                min_dist = self.dist[ix][iy]
                min_dist_city1 = ix
                min_dist_city2 = iy

        perm.append(min_dist_city1)
        perm.append(min_dist_city2)
        temp.remove(min_dist_city1)
        temp.remove(min_dist_city2)

        # Find the next city which is closest to one of the cities in the current tour
        while len(temp) > 0:
            if len(temp) == 1:
                next_city = temp[0]
            else:
                # Find the next city which is the closest to an arbitrary city in the current tour
                for t in perm:
                    min_dist_city = self.dist[t][temp[0]]
                    for city in temp[1:]:
                        if min_dist_city > self.dist[t][city]:
                            min_dist_city = self.dist[t][city]
                            next_city = city

            # Find an edge in the current tour that satisfies the condition:
            # distance[city1][next_city] + distance[next_city][city2] - distance[city1][city2]
            if len(perm) == 2:
                perm = [perm[0], next_city, perm[1]]
            else:
                min_dist_to_add = self.dist[perm[0]][next_city] + \
                                  self.dist[next_city][perm[1]] - self.dist[perm[0]][perm[1]]

                idx_add = 1  # Start from the second element in the current tour because of the starting min distance
                idx_first = 0  # This is used to find the index of the first edge
                idx_second = 1  # This is used to find the index of the second edge

                # Loop to find the closest pair
                while idx_add < len(perm):
                    if min_dist_to_add > self.dist[perm[idx_add]][next_city] + \
                            self.dist[next_city][perm[(idx_add + 1) % len(perm)]] - \
                            self.dist[perm[idx_add]][perm[(idx_add + 1) % len(perm)]]:

                        min_dist_to_add = self.dist[perm[idx_add]][next_city] + \
                            self.dist[next_city][perm[(idx_add + 1) % len(perm)]] - \
                            self.dist[perm[idx_add]][perm[(idx_add + 1) % len(perm)]]

                        idx_first = idx_add  # First city of the pair
                        idx_second = (idx_add + 1) % len(perm)  # Second city of the pair
                    idx_add += 1

                if idx_first == len(perm) - 1:  # In case the pair is the last and the first city
                    perm.append(next_city)
                else:
                    perm_split_first = perm[:idx_second]
                    perm_split_first.append(next_city)
                    perm_split_second = perm[idx_second:]
                    perm = perm_split_first + perm_split_second

            # Prints for testing
            print("Temp array:")
            print(temp)
            print("Next city:")
            print(next_city)
            temp.remove(next_city)
            print("Permutations: ")
            print(perm)

        # Set the new permutation
        self.perm = np.copy(np.array(perm))
