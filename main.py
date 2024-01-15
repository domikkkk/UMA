import qlearning
import evolution
import numpy as np


easy = lambda x: x[0]*x[0]+x[1]*x[1]


def main():
    agent = qlearning.QLearning_evolution([-5,0,5],[-0.1,-0,0.1],easy)
    agent.fit()

if __name__=="__main__":
    main()