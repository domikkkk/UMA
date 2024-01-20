import qlearning
import evolution
import numpy as np


easy = lambda x: x[0]*x[0]+x[1]*x[1]


def main():
    np.random.seed(5)
    agent = qlearning.QLearning_evolution([-5,0,5],[-0.1,-0,0.1],easy, alpha=0.2,epsilon=0.5,gamma=0.1)
    agent.fit(episodes=100)

    c = 0
    for s in range(50):
        # test results
        print("test")
        agent.reset(seed=s)
        agent.episode(learn=False,steps=25)
        no_q=agent._env.mean_and_deviation()[0]
        #print(agent._env.mean_and_deviation())

        # manual steps without qlearning test
        agent.reset(seed=s)
        for episode in range(25):
            agent.get_environment().step()
        #print(agent._env.mean_and_deviation())
        with_q=agent._env.mean_and_deviation()[0]

        print(with_q<no_q)
        c+=1 if with_q<no_q else 0
    print(c)

if __name__=="__main__":
    main()