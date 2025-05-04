from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Create start variables for jobs
start_a = model.NewIntVar(1, 10, 'start_a')
start_b = model.NewIntVar(1, 10, 'start_b')
start_c = model.NewIntVar(1, 10, 'start_c')

durations = [3, 2, 2]

# End times
end_a = model.NewIntVar(0, 15, 'end_a')
end_b = model.NewIntVar(0, 15, 'end_b')
end_c = model.NewIntVar(0, 15, 'end_c')

# Constraints: end = start + duration
model.Add(end_a == start_a + durations[0])
model.Add(end_b == start_b + durations[1])
model.Add(end_c == start_c + durations[2])

interval_a = model.NewIntervalVar(start_a, durations[0], end_a, 'interval_a')
interval_b = model.NewIntervalVar(start_b, durations[1], end_b, 'interval_b')
interval_c = model.NewIntervalVar(start_c, durations[2], end_c, 'interval_c')

# No overlap constraint
model.AddNoOverlap([interval_a, interval_b, interval_c])

# Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
    print("A:", solver.Value(start_a))
    print("B:", solver.Value(start_b))
    print("C:", solver.Value(start_c))
else:
    print("No feasible solution found.")
