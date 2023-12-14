





























































































































































































































































































































































































































































































































































#minmax
def getMinMax(low, high, arr):
	arr_max = arr[low]
	arr_min = arr[low]

	if low == high:
		arr_max = arr[low]
		arr_min = arr[low]
		return (arr_max, arr_min)

	elif high == low + 1:
		if arr[low] > arr[high]:
			arr_max = arr[low]
			arr_min = arr[high]
		else:
			arr_max = arr[high]
			arr_min = arr[low]
		return (arr_max, arr_min)
	else:

		mid = int((low + high) / 2)
		arr_max1, arr_min1 = getMinMax(low, mid, arr)
		arr_max2, arr_min2 = getMinMax(mid + 1, high, arr)

	return (max(arr_max1, arr_max2), min(arr_min1, arr_min2))


arr = [1000, 11, 445, 1, 330, 3000]
high = len(arr) - 1
low = 0
arr_max, arr_min = getMinMax(low, high, arr)
print('Minimum element is ', arr_min)
print('nMaximum element is ', arr_max)





#fractional knap sack
class Item: 
    def __init__ (self, profit, weight): 
        self.profit = profit 
        self.weight = weight 
def fractionalKnapsack(W, arr): 
    arr.sort(key=lambda x: (x.profit/x.weight), reverse=True) 
    finalvalue = 0.0 
    for item in arr: 
        if item.weight <= W: 
            W -= item.weight 
            finalvalue += item.profit 
        else: 
            finalvalue += item.profit * W / item.weight 
            break 
    return finalvalue 
W = int(input("Enter the maximum capacity of the knapsack")) 
arr = [] 
n = int(input("Enter the number of items")) 
for i in range(n): 
    print("Enter the profit of item ",i+1) 
    x = int(input()) 
    print("Enter the weight of item ",i+1) 
    y = int(input()) 
    arr.append(Item(x,y)) 
    max_val = fractionalKnapsack(W, arr) 
print("Maximum profit is ",max_val) 

#0/1 knapsack


def knapSack(W, wt, val, n): 
    K = [[0 for x in range(W + 1)] for x in range(n + 1)] 
  
    # Build table K[][] in bottom up manner 
    for i in range(n + 1): 
        for w in range(W + 1): 
            if i == 0 or w == 0: 
                K[i][w] = 0
            elif wt[i-1] <= w: 
                K[i][w] = max(val[i-1] 
                              + K[i-1][w-wt[i-1]], 
                              K[i-1][w]) 
            else: 
                K[i][w] = K[i-1][w] 
  
    return K[n][W]

profit = [60, 100, 120] 
weight = [10, 20, 30] 
W = 50
n = len(profit) 
print(knapSack(W, weight, profit, n)) 


#job scheduling

def printJobScheduling(arr, t): 
    n = len(arr) 
    for i in range(n): 
        for j in range(n - 1 - i): 
            if arr[j][2] < arr[j + 1][2]: 
                arr[j], arr[j + 1] = arr[j + 1], arr[j] 
    result = [False] * t 
    job = ['-1'] * t 
    for i in range(len(arr)): 
        for j in range(min(t - 1, arr[i][1] - 1), -1, -1): 
            if result[j] is False: 
                result[j] = True 
                job[j] = arr[i][0] 
                break 
    print(job) 
a=[] 
n=int(input("Enter number of jobs:")) 
maxi=0 
print("Enter job name,deadlines and profits:") 
for i in range(n): 
    x,y,z=map(eval,input().split()) 
    maxi=max(maxi,y) 
    a.append([x,y,z]) 
print("Following is maximum profit sequence of jobs") 
printJobScheduling(a, maxi)


#n queens

global N
N = 4
def printSolution(board):
	for i in range(N):
		for j in range(N):
			if board[i][j] == 1:
				print("Q",end=" ")
			else:
				print(".",end=" ")
		print()



def isSafe(board, row, col):


	for i in range(col):
		if board[row][i] == 1:
			return False


	for i, j in zip(range(row, -1, -1),
					range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	for i, j in zip(range(row, N, 1),
					range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	return True


def solveNQUtil(board, col):


	if col >= N:
		return True
	for i in range(N):

		if isSafe(board, i, col):


			board[i][col] = 1


			if solveNQUtil(board, col + 1) == True:
				return True

			board[i][col] = 0
	return False

def solveNQ():
	board = [[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0]]

	if solveNQUtil(board, 0) == False:
		print("Solution does not exist")
		return False

	printSolution(board)
	return True



if __name__ == '__main__':
	solveNQ()



#longest common sub
def lcs(self,x,y,s1,s2):
        a=[[0 for i in range(y+1)] for j in range(x+1)]
        for i in range(x+1):
            for j in range(y+1):
                if(i==0):a[0][j]=0;
                elif(j==0):a[i][0]=0;
                elif(s1[i-1]==s2[j-1]):a[i][j]=a[i-1][j-1]+1
                else:a[i][j]=max(a[i-1][j],a[i][j-1])
        return a[x][y]

#matrix chain mul
def mcm(a,i,j):
    if(i==j):return 0
    if(dp[i][j]!=-1):return dp[i][j]
    dp[i][j]=2**30
    for k in range(i,j):
        dp[i][j]=min(dp[i][j],mcm(a,i,k)+mcm(a,k+1,j)+a[i]*a[k+1]*a[j+1])
    return dp[i][j]
n=int(input())
a=[int(i) for i in input().split()]
dp=[[-1 for i in range(n)] for j in range(n)]
print(mcm(a,0,n-2))



#huffman


import heapq
from collections import defaultdict
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq
def build_huffman_tree(freq_dict):
    heap = [Node(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]
def build_huffman_codes(root, current_code, huffman_codes):
    if root is None:
        return
    if root.char is not None:
        huffman_codes[root.char] = current_code
    build_huffman_codes(root.left, current_code + '0', huffman_codes)
    build_huffman_codes(root.right, current_code + '1', huffman_codes)
def huffman_encoding(data):
    if not data:
        return "", None
    freq_dict = defaultdict(int)
    for char in data:
        freq_dict[char] += 1
    root = build_huffman_tree(freq_dict)
    huffman_codes = {}
    build_huffman_codes(root, '', huffman_codes)
    encoded_data = ''.join(huffman_codes[char] for char in data)
    return encoded_data, root
def huffman_decoding(encoded_data, root):
    if not encoded_data:
        return ""
    decoded_data = ""
    current = root
    for bit in encoded_data:
        if bit == '0':
            current = current.left
        else:
            current = current.right
        if current.char is not None:
            decoded_data += current.char
            current = root
    return decoded_data

data = "this is an example for huffman encoding"

encoded_data, tree = huffman_encoding(data)
print(f"Encoded data: {encoded_data}")

decoded_data = huffman_decoding(encoded_data, tree)
print(f"Decoded data: {decoded_data}")




#allpairs, floyd

def floyd_warshall(graph):
    n = len(graph)


    distance = [row[:] for row in graph]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][k] != float('inf') and distance[k][j] != float('inf'):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
        print("\nThrough ",k+1," : ")
        for row in distance:
            print(row)

    return distance


n=int(input("Enter number of vertices: "))
adjacency_matrix= [[float('inf')] * n for _ in range(n)]

print("Enter the graph (enter 'inf' for infinity):")
for i in range(n):
        for j in range(n):
            while True:
                try:
                    value = input(f"Weight from vertex {i+1} to {j+1}: ")
                    if value.lower() == 'inf':
                        adjacency_matrix[i][j] = float('inf')
                    else:
                        adjacency_matrix[i][j] = int(value)
                    break
                except ValueError:
                    print("Invalid input. Please enter an integer or 'inf'.")
'''adjacency_matrix = [
        [0, 3, float('inf'), 7],
        [8, 0, 2, float('inf')],
        [5, float('inf'), 0,1],
        [2, float('inf'), float('inf'), 0, ]
    ]'''

print("Initial Distance Matrix:")
for row in adjacency_matrix:
        print(row)
result = floyd_warshall(adjacency_matrix)
print("\nFinal Distance Matrix:")
for row in result:
        print(row)


#bellman
def bellman_ford(adj_matrix, source):
    vertices = len(adj_matrix)
    dist = [float('inf')] * vertices
    dist[source] = 0
    for _ in range(vertices - 1):
        print("\n\nRelation iteration ",_+1,":")
        for u in range(vertices):
            for v in range(vertices):
                if adj_matrix[u][v] != 0: 
                    if dist[u] + adj_matrix[u][v] < dist[v]:
                        dist[v] = dist[u] + adj_matrix[u][v]
                    print("After relaxation of (",u+1,",",v+1,") edge: ",dist)
    for u in range(vertices):
        for v in range(vertices):
            if adj_matrix[u][v] != 0: 
                if dist[u] + adj_matrix[u][v] < dist[v]:
                    raise ValueError("Graph contains a negative cycle")

    return dist

'''graph_matrix = [
    [0, -1, 4, 0, 0],
    [0, 0, 3, 2, 2],
    [0, 0, 0, 0, 0],
    [0, 1, 5, 0, 0],
    [0, 0, 0, -3, 0]
]'''
N=int(input())
G=[]
print("Enter adjacency matrix:")
for _ in range(N):
    row=list(map(int,input().split()))
    G.append(row)
source_vertex = 0
result = bellman_ford(G, source_vertex)
for i in range(len(result)):
        print(f"Vertex {i + 1}: Distance = {result[i]}")

#krushkals
def kruskal_algo(n, graph):
    def find(component):
        if parent[component] == component:
            return component
        temp = find(parent[component])
        parent[component] = temp
        return temp

    def union(vertex1, vertex2):
        parent_of_vertex1 = find(vertex1)
        parent_of_vertex2 = find(vertex2)

        if parent_of_vertex1 == parent_of_vertex2:
            return True

        if rank[parent_of_vertex1] > rank[parent_of_vertex2]:
            parent[parent_of_vertex2] = parent_of_vertex1
        elif rank[parent_of_vertex1] < rank[parent_of_vertex2]:
            parent[parent_of_vertex1] = parent_of_vertex2
        else:
            parent[parent_of_vertex1] = parent_of_vertex2
            rank[parent_of_vertex2] += 1

        return False

    print("Minimum Spanning Tree is :-")
    print("V1", "V2", "Wt")

    ans = 0
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    edges.sort(key=lambda x: x[2])

    parent = [i for i in range(n)]
    rank = [1] * n

    for edge in edges:
        vertex1, vertex2, weight = edge
        flag = union(vertex1, vertex2)

        if not flag:
            print(vertex1, vertex2, weight)
            ans += weight

    return ans


adjacency_matrix = [[0, 28, 0, 0 ,0, 10, 0],
[28, 0, 16, 0 ,0 ,0 ,14],
[0 ,16 ,0 ,12 ,0 ,0 ,0],
[0, 0 ,12 ,0 ,22 ,0 ,18],
[0, 0 ,0 ,22 ,0 ,25, 24],
[10, 0 ,0 ,0 ,25 ,0 ,0],
[0 ,14 ,0 ,18 ,24 ,0 ,0]]

ans = kruskal_algo(7, adjacency_matrix)
print("The min cost is",ans)

#prims mst
INF = 9999999
#N = int(input("Enter number of vertices: "))

N=7
'''
N=int(input())
G=[]
print("Enter adjacency matrix:")
for _ in range(N):
    row=list(map(int,input().split()))
    G.append(row)'''

G=[[0, 28, 0, 0 ,0, 10, 0],
[28, 0, 16, 0 ,0 ,0 ,14],
[0 ,16 ,0 ,12 ,0 ,0 ,0],
[0, 0 ,12 ,0 ,22 ,0 ,18],
[0, 0 ,0 ,22 ,0 ,25, 24],
[10, 0 ,0 ,0 ,25 ,0 ,0],
[0 ,14 ,0 ,18 ,24 ,0 ,0]]
visited = [False]*N
no_edge = 0
visited[0] = True
l=[]
cost=0
row=[]
while (no_edge < N - 1):
    minimum = INF
    a = 0
    b = 0
    for m in range(N):
        if visited[m]:
            for n in range(N):
                    
                if ((not visited[n]) and G[m][n]):
                    row.append([m+1,n+1,G[m][n]])
                    if minimum > G[m][n]:
                        minimum = G[m][n]
                        a = m
                        b = n
    print("List of nodes connected to ",a+1)
    for i in row:
        if i[0]==a+1:
            print(i[1],"-",i[2])
    row=[]
    cost+=G[a][b]
    l.append(str(a+1) + "-" + str(b+1) + ":" + str(G[a][b]))
    print("Minimum edge is ",b+1,"-",G[a][b],"\n")
    visited[b] = True
    no_edge += 1
print("Edge : Weight")
for i in l:
    print(i)
print("Cost of Tree : ",cost)



#dijkstra , sssp
import heapq
def dijkstra(graph, start):
    num_vertices = len(graph)
    distance = [float('inf')] * num_vertices
    predecessor = [None] * num_vertices
    visited = [False] * num_vertices

    distance[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        dist_u, u = heapq.heappop(priority_queue)

        if visited[u]:
            continue

        visited[u] = True

        for v, weight in enumerate(graph[u]):
            if not visited[v] and weight != 0:
                new_distance = distance[u] + weight

                if new_distance < distance[v]:
                    distance[v] = new_distance
                    predecessor[v] = u
                    heapq.heappush(priority_queue, (distance[v], v))

    return distance, predecessor

def print_results(distance, predecessor, start):
    print("\nFinal Results:")
    for i in range(len(distance)):
        print(f"Vertex {i + 1}: Shortest Distance = {distance[i]}, Predecessor = {predecessor[i] + 1 if predecessor[i] is not None else None}")
    print(f"\nShortest Paths from Vertex {start + 1} to other vertices:")
    for i in range(len(distance)):
        path = [i + 1]
        current = i
        while predecessor[current] is not None:
            path.insert(0, predecessor[current] + 1)
            current = predecessor[current]
        print(f"To Vertex {i + 1}: {path}")
n = int(input("Enter the number of vertices: "))
start_vertex = int(input("Enter the start vertex (1 to N): ")) - 1
'''graph=[]
print("Enter adjacency matrix:")
for _ in range(n):
    row=list(map(int,input().split()))
    graph.append(row)'''
graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
               [4, 0, 8, 0, 0, 0, 0, 11, 0],
               [0, 8, 0, 7, 0, 4, 0, 0, 2],
               [0, 0, 7, 0, 9, 14, 0, 0, 0],
               [0, 0, 0, 9, 0, 10, 0, 0, 0],
               [0, 0, 4, 14, 10, 0, 2, 0, 0],
               [0, 0, 0, 0, 0, 2, 0, 1, 6],
               [8, 11, 0, 0, 0, 0, 1, 0, 7],
               [0, 0, 2, 0, 0, 0, 6, 7, 0]
               ]
print("\nGraph:")
for row in graph:
        print(row)
distances, predecessors = dijkstra(graph, start_vertex)
print_results(distances, predecessors, start_vertex)
