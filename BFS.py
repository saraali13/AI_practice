def bfs(graph, start, goal):
    visited = []  # List to track visited nodes
    queue = [[start]]  # Queue stores paths instead of single nodes

    while queue:
        path = queue.pop(0)  # Get the first path in the queue
        node = path[-1]  # Last node in the current path
        print(node, end=" ")
        if node == goal:  # Stop if goal is found
            print("\nBFS Path:", " -> ".join(path))
            return path  # Return the successful path

        if node not in visited:
            visited.append(node)  # Mark the node as visited

            for neighbour in graph[node]:  # Explore neighbors
                new_path = list(path)  # Create a copy of the current path
                new_path.append(neighbour)  # Add neighbor to the path
                queue.append(new_path)  # Add the new path to the queue

    print("Goal not found!")
    return None  # Return None if no path exists


# Graph Definition
Graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Example Execution
bfs(Graph, 'A', 'F')
