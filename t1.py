import numpy as np

p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n = np.array(p)

q = 0
t = 3

print(n[:0])
print(n[q:t+1])
print(n[t+1:])
m = np.concatenate((n[:q], np.flip(n[q:t+1]), n[t+1:]))
print(m)

j = np.arange(5, 12)
print(p.count(7) > 0)

for i in range(5):
    print(i)
    n = np.delete(n, np.argwhere(n == i))
print(n)

q = np.array([0])

f = 1
while f < len(p):
    print(p[f % len(p)])
    print(p[(f+1) % len(p)])
    f += 1
print(p[1:3] + p[3:6])