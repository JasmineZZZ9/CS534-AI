# Mingjie Zeng - 671222265 - Assignment 1
# Ex 2.9
# 1. agent : left, right, suck
# 2. environment : A and B, clean or dirty
# 3. performance score for each & average score

from modules.agents import *

# define a simple reflex agent for the vacuum-cleaner

def SimpleReflexAgent():

    def program(percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        elif location == loc_A:
            return 'Right'
        elif location == loc_B:
            return 'Left'

    return Agent(program)


# this is an environment that the agent's initial location is loc_A
# environment is defined in agents.py
agentA = SimpleReflexAgent()
environment_locA = TrivialVacuumEnvironment_locA()

# this is an environment that the agent's initial location is loc_B
# environment is defined in agents.py
agentB = SimpleReflexAgent()
environment_locB = TrivialVacuumEnvironment_locB()

# variable sum is to sum up overall scores
sum_score = 0
# variable situation is to count the number of initial configurations
situation = 0

# there are 4 types of dirt distributions
# {loc_A:'Dirty',loc_B:'Dirty'}
# {loc_A:'Clean',loc_B:'Dirty'}
# {loc_A:'Dirty',loc_B:'Clean'}
# {loc_A:'Clean',loc_B:'Clean'}

for count_initialAgentLoc in range(0, 2):
    # environment_locA
    if count_initialAgentLoc == 0: 
        for count_initialDirtConfig in range(0, 4):

            # {loc_A:'Dirty',loc_B:'Dirty'}
            if count_initialDirtConfig == 0:
                # dirt configure
                environment_locA.status = {loc_A:'Dirty',loc_B:'Dirty'}
                print("initial dirt configuration is {}".format(environment_locA.status))
                # run this environment and agent
                environment_locA.add_thing(agentA)
                environment_locA.run(5)
                print("initial agent location is {}".format(agentA.location))
                print("performance score is {}".format(agentA.performance))
                print("-----------------------------------------------------------------")
                sum_score += agentA.performance
                situation += 1
                environment_locA.delete_thing(agentA)

            # {loc_A:'Clean',loc_B:'Dirty'}
            elif count_initialDirtConfig == 1:
                # dirt configure
                environment_locA.status = {loc_A:'Clean',loc_B:'Dirty'}
                print("initial dirt configuration is {}".format(environment_locA.status))
                # run this environment and agent
                environment_locA.add_thing(agentA)
                environment_locA.run(5)
                print("initial agent location is {}".format(agentA.location))
                print("performance score is {}".format(agentA.performance))
                print("-----------------------------------------------------------------")
                sum_score += agentA.performance
                situation += 1
                environment_locA.delete_thing(agentA)
            
            # {loc_A:'Dirty',loc_B:'Clean'}
            elif count_initialDirtConfig == 2:
                # dirt configure
                environment_locA.status = {loc_A:'Dirty',loc_B:'Clean'}
                print("initial dirt configuration is {}".format(environment_locA.status))
                # run this environment and agent
                environment_locA.add_thing(agentA)
                environment_locA.run(5)
                print("initial agent location is {}".format(agentA.location))
                print("performance score is {}".format(agentA.performance))
                print("-----------------------------------------------------------------")
                sum_score += agentA.performance
                situation += 1
                environment_locA.delete_thing(agentA)
            
            # {loc_A:'Clean',loc_B:'Clean'}
            elif count_initialDirtConfig == 3:
                # dirt configure
                environment_locA.status = {loc_A:'Clean',loc_B:'Clean'}
                print("initial dirt configuration is {}".format(environment_locA.status))
                # run this environment and agent
                environment_locA.add_thing(agentA)
                environment_locA.run(5)
                print("initial agent location is {}".format(agentA.location))
                print("performance score is {}".format(agentA.performance))
                print("-----------------------------------------------------------------")
                sum_score += agentA.performance
                situation += 1
                environment_locA.delete_thing(agentA)
    # environment_locB
    elif count_initialAgentLoc == 1: 
        for count_initialDirtConfig in range(0, 4):

            # {loc_A:'Dirty',loc_B:'Dirty'}
            if count_initialDirtConfig == 0:
                # dirt configure
                environment_locB.status = {loc_A:'Dirty',loc_B:'Dirty'}
                print("initial dirt configuration is {}".format(environment_locB.status))
                # run this environment and agent
                environment_locB.add_thing(agentB)
                environment_locB.run(5)
                print("initial agent location is {}".format(agentB.location))
                print("performance score is {}".format(agentB.performance))
                print("-----------------------------------------------------------------")
                sum_score += agentB.performance
                situation += 1
                environment_locB.delete_thing(agentB)
            
            # {loc_A:'Clean',loc_B:'Dirty'}
            elif count_initialDirtConfig == 1:
                # dirt configure
                environment_locB.status = {loc_A:'Clean',loc_B:'Dirty'}
                print("initial dirt configuration is {}".format(environment_locB.status))
                # run this environment and agent
                environment_locB.add_thing(agentB)
                environment_locB.run(5)
                print("initial agent location is {}".format(agentB.location))
                print("performance score is {}".format(agentB.performance))
                print("-----------------------------------------------------------------")
                sum_score += agentB.performance
                situation += 1
                environment_locB.delete_thing(agentB)
            
            # {loc_A:'Dirty',loc_B:'Clean'}
            elif count_initialDirtConfig == 2:
                # dirt configure
                environment_locB.status = {loc_A:'Dirty',loc_B:'Clean'}
                print("initial dirt configuration is {}".format(environment_locB.status))
                # run this environment and agent
                environment_locB.add_thing(agentB)
                environment_locB.run(5)
                print("initial agent location is {}".format(agentB.location))
                print("performance score is {}".format(agentB.performance))
                print("-----------------------------------------------------------------")
                sum_score += agentB.performance
                situation += 1
                environment_locB.delete_thing(agentB)
            
            # {loc_A:'Clean',loc_B:'Clean'}
            elif count_initialDirtConfig == 3:
                # dirt configure
                environment_locB.status = {loc_A:'Clean',loc_B:'Clean'}
                print("initial dirt configuration is {}".format(environment_locB.status))
                # run this environment and agent
                environment_locB.add_thing(agentB)
                environment_locB.run(5)
                print("initial agent location is {}".format(agentB.location))
                print("performance score is {}".format(agentB.performance))
                print("-----------------------------------------------------------------")
                sum_score += agentB.performance
                situation += 1
                environment_locB.delete_thing(agentB)


print("There are {} configurations.".format(situation))
print("The overall average score is {}".format(sum_score/situation))








