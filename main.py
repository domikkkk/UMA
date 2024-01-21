import qlearning
import evolution
import numpy as np
from cec2017.functions import f3

np.set_printoptions(precision=4,suppress=True)

easy = lambda x: x[0]*x[0]+x[1]*x[1]


def main():
    np.random.seed(2)
    agent = qlearning.QLearning_evolution([1,0,-1],
                                          [-0.1,0,0.1],
                                          f3,
                                          alpha=0.01,
                                          epsilon=0.9,
                                          epsilon_min=0.3,
                                          epsilon_decay=0.99,
                                          gamma=0.6,
                                          state_size=(5,5),
                                          population_size=20)
    agent.fit(episodes=250, steps_per_episode=100)

    c = 0
    for s in range(50):
        # test results
        agent.reset(seed=s)
        agent.episode(learn=False,steps=25, verbose=True)
        print(f"seed #{s} With Q learning:")
        with_q=agent._env.mean_and_deviation()[0]

        # manual steps without qlearning test
        agent.reset(seed=s)
        print(f"seed #{s} just evolution:")
        for step in range(25):
            agent._env.step()
            print(agent._env.mean_and_deviation(), agent._env.population_size, agent._env.sigma)
        no_q=agent._env.mean_and_deviation()[0]

        c+=1 if with_q<no_q else 0
    print(f"Q learning helped in {c}/50 cases")

if __name__=="__main__":
    main()