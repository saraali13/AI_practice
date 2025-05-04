from constraint import Problem
# pip install python-constraint

def map_coloring():
    problem = Problem()

    # Ndes
    states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # Domain: colors
    colors = ['red', 'green', 'blue']
    #edges 
    edges = [("A", "B"), ("B", "C"), ("C", "G"), ("D", "E"), ("E", "F"),("F","A")]
    # Add variables and domains
    problem.addVariables(states, colors)

    # Add constraints (adjacent regions must have different colors)
    for node1, node2 in edges:
        problem.addConstraint(lambda x, y: x != y, (node1, node2))

    # Solve the problem (list of all possible sol)
    solutions = problem.getSolutions()
    return solutions

print(map_coloring())
