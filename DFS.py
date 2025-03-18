def dfs(graph, start, goal):
    visited = [] # List for visited nodes
    stack = [[start]] # Initialize stack

    while stack:
        path = stack.pop() # LIFO: Pop from top
        node=path[-1]
        print(node, end=" ")
        if node == goal: # Stop if goal is found
            print("\nDFS Path:", " -> ".join(path))
            return path

        for neighbour in reversed(graph[node]): # Reverse to maintain correct order
            if neighbour not in visited:
                new_path = list(path)  # Create a copy of the current path
                new_path.append(neighbour)
                visited.append(neighbour)
                stack.append(new_path)

Graph = {
'A': ['B', 'C'],
'B': ['D', 'E'],
'C': ['F', 'G'],
'D': ['H'],
'E': [],
'F': ['I'],
'G': [],
'H': [],
'I': []
}
start_node = 'A'
goal_node = 'I'
# Run DFS
print("\nFollowing is the Depth-First Search (DFS):")
dfs(Graph, start_node, goal_node)
