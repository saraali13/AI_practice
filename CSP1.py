from ortools.sat.python import cp_model

def solve_csp_example():
    model = cp_model.CpModel() # set the model

    # Variables
    x = model.NewIntVar(0, 10, 'x')
    y = model.NewIntVar(0, 10, 'y')

    # Constraints
    model.Add(x + y == 7)
    model.Add(x - y == 3)
    model.Add(x>y)
   # model.Add(x<y)
    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print(f"x = {solver.Value(x)}")
        print(f"y = {solver.Value(y)}")
    else:
        print("No optimal solution")

solve_csp_example()
