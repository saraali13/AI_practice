from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the network structure
model = DiscreteBayesianNetwork([
    ('Intelligence', 'Grade'),
    ('StudyHours', 'Grade'),
    ('Difficulty', 'Grade')
])

# Define CPDs
cpd_intelligence = TabularCPD('Intelligence', 2, [[0.7], [0.3]], state_names={'Intelligence': ['High', 'Low']})
cpd_studyhours = TabularCPD('StudyHours', 2, [[0.6], [0.4]], state_names={'StudyHours': ['Sufficient', 'Insufficient']})
cpd_difficulty = TabularCPD('Difficulty', 2, [[0.5], [0.5]], state_names={'Difficulty': ['Hard', 'Easy']})

cpd_grade = TabularCPD(
    variable='Grade',
    variable_card=3,
    values=[
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3],
        [0.08, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3],
        [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.3, 0.4],
    ],
    evidence=['Intelligence', 'StudyHours', 'Difficulty'],
    evidence_card=[2, 2, 2],
    state_names={
        'Grade': ['A', 'B', 'C'],
        'Intelligence': ['High', 'Low'],
        'StudyHours': ['Sufficient', 'Insufficient'],
        'Difficulty': ['Hard', 'Easy']
    }
)

# Add CPDs and check model
model.add_cpds(cpd_intelligence, cpd_studyhours, cpd_difficulty, cpd_grade)
model.check_model()

# Create inference engine
infer = VariableElimination(model)

result1 = infer.query(['Grade'], evidence={
    'Intelligence': 'Low',
    'StudyHours': 'Sufficient',
    'Difficulty': 'Easy'
})
print(result1)
#P(Grade=A) = 0.4
#P(Grade=B) = 0.3
#P(Grade=C) = 0.3

result2 = infer.query(['Grade'], evidence={
    'StudyHours': 'Insufficient',
    'Difficulty': 'Hard'
})
print(result2)

result3 = infer.query(['Intelligence'], evidence={'Grade': 'A'})
print(result3)

result4 = infer.query(['Difficulty'], evidence={
    'Grade': 'C',
    'StudyHours': 'Insufficient'
})
print(result4)

