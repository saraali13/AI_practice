class Environment:
    def __init__(self, initial_state):
        self.initial_state = initial_state #Initial state could be fixed or random
    def get_percept(self):
     #initial condition of environment , that would be perceived by agent
     pass
class SimpleReflexAgent:
    def __init__(self):
        pass
    def act(self, percept):
      #Determine action based on the initial percept
      pass

def run_agent(agent, environment):
 # The agent reacts to the initial stimulus/Percept
    percept = environment.get_percept()
    action = agent.act(percept)
    print(f"Percept: {percept}, Action: {action}")

agent = SimpleReflexAgent()
environment = Environment(initial_state=0)
run_agent(agent, environment)
