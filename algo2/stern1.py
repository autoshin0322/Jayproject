import math

Zeile = input()

vertices = int(Zeile.split()[0])
edges = int(Zeile.split()[1])

# herstelle leere Matrix
matrix = []
for i in range(0, vertices):
    low = []
    for j in range(0, vertices):
        if i == j:
            low.append(0)
        else:
            low.append(math.inf)
    matrix.append(low)

# zuerst Kante (u,v) hinzufuegen
for i in range(0, edges):
    SSAP = input()
    u, v, w = int(SSAP.split()[0]), int(SSAP.split()[1]), int(SSAP.split()[2])
    matrix[u-1][v-1] = w

# finde Kante (u,w) - ... - (w,v)
for low in range(0, vertices):
    for col in range(0, vertices):
        if matrix[low][col] == math.inf:
            list = []
            for j in range(0, vertices):
                if (matrix[low][j] != math.inf) and (matrix[j][col] != math.inf):
                    value = matrix[low][j] + matrix[j][col]
                    list.append(value)
                    matrix[low][col] = min(list)

# print Ereignis
for i in range(0, vertices):
    for j in range(0, vertices):
        print(matrix[i][j], end=" ")
    print()


# implementation of the Floyd-Warshall algorithm
import math

n, m = map(int, input().split())

if n != 0 and m != 0:
    
    # populate weight lookup table
    
    dist = [[math.inf for _ in range(n)] for _ in range(n)]
    
    for i in range(m): 
        u, v, w = map(int, input().split())
        dist[u-1][v-1] = w
        
        
    # Floyd-Warshall
                
    for r in range(n):
        for u in range(n):
            for v in range(n):
                if dist[u][v] > dist[u][r] + dist[r][v]:
                    dist[u][v] = dist[u][r] + dist[r][v]
                if u==v:
                    dist[u][v] = 0
                    
                
    print("\n".join
                (" ".join(
                        str(dist[i][j]) 
                    for j in range(n)) 
                for i in range(n)))