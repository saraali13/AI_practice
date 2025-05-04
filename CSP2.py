from ortools.sat.python import cp_model
from typing import List

class SolPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, var: List[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.vars = var
        self._sol_count = 0

    def on_solution_callback(self):
        self._sol_count += 1
        for var in self.vars:
            print(f"{var.Name()} = {self.Value(var)}")
        print()

    def solution_count(self):
        return self._sol_count

# Model definition
model = cp_model.CpModel()
x = model.NewIntVar(0, 3, "x")
y = model.NewIntVar(0, 3, "y")
z = model.NewIntVar(0, 3, "z")

model.Add(x < y)
# model.AddAllDifferent([x, y, z])  # Uncomment if needed

# Solver and solution printer
solver = cp_model.CpSolver()
solution_printer = SolPrinter([x, y, z])
solver.parameters.enumerate_all_solutions = True

status = solver.Solve(model, solution_printer)

# Output
print(f"Status: {solver.StatusName(status)}")
print(f"Number of solutions found: {solution_printer.solution_count()}")
