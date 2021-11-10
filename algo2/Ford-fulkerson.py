# Eingabe: Flussnetzwerk
# Ausgabe: maximalen Fluss zwischen zwei Knoten des Netzwerkes berechnen.
# mithilfe Ford-Fulkerson Algorithmus

"""
Eingabe Bsp1:

4 5
1 2 20
2 3 30
2 4 10
1 3 10
3 4 20

Bsp2:

10 20
1 9 27
1 8 10
1 7 17
2 7 7
2 4 34
2 9 30
2 6 14
3 5 21
4 7 4
5 10 21
6 2 31
6 5 13
7 8 29
7 3 44
8 10 42
8 4 39
8 7 42
9 6 16
9 3 23
9 10 0
"""

n, m = map(int, input().split())

graph = list()

for i in range(0, n):
    graph.append([])

# add edges with weight to each knote
for i in range(0, m):
    SSAP = input()
    u, v, c = int(SSAP.split()[0]), int(SSAP.split()[1]), int(SSAP.split()[2])
    graph[u-1].append((v,c))

# BFS 

"""
# calculate Fluss of each Knote
maximum = list()
for i in range(len(graph)):
    sum = 0
    for j in range(len(graph[i-1])):
        sum += int(graph[i-1][j-1][1])
    maximum.append(sum)
"""
print(graph[0][1][0])