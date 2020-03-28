import pandas as pd
import timeit
import graph

g = graph.Graph(-1, "cities50")

g.swapHeuristic()

print(g.tourValue())

# g1 = graph.Graph(6, "sixnodes")
# print(g1.tourValue())
#
# g1.swapHeuristic()
# print(g1.tourValue())

g2 = graph.Graph(-1, "cities50")
g2.swapHeuristic()
g2.TwoOptHeuristic()
print(g2.tourValue())

g3 = graph.Graph(-1, "cities25")
g3.Greedy()
print(g3.tourValue())
print(timeit.timeit(g3.Greedy, number=10) / 10)
print(timeit.timeit(g2.swapHeuristic, number=10) / 10)
print(timeit.timeit(g2.TwoOptHeuristic, number=10) / 10)

g4 = graph.Graph(-1, "cities25")
print(pd.DataFrame(g4.dist))
g4.nearest_insertion()
print(g4.tourValue())