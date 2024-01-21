import sys

try:
    sys.path.append('./')
    from cec2017.functions import *
except Exception:
    sys.path.append('../')
    from cec2017.functions import *
from qlearning import QLearning_evolution


def learn(functions_to_learn, actions_P, actions_M, p_size):
    Q = QLearning_evolution(actions_P,
                            actions_M,
                            None,
                            alpha=0.01,
                            epsilon=0.9,
                            epsilon_min=0.2,
                            epsilon_decay=0.995,
                            gamma=0.6,
                            state_size=(5,5),
                            population_size=p_size)
    for f in functions_to_learn:
        Q.objective = f
        Q.fit(episodes=1000//len(functions_to_learn), steps_per_episode=100)
    print("Done")
    return Q


def test(Q: QLearning_evolution, f):
    Q.objective = f
    c = 0
    for s in range(50):
        # test results
        Q.reset(seed=s)
        print(f"seed #{s} With Q learning:")
        Q.episode(learn=False,steps=25, verbose=True)
        with_q=Q._env.mean_and_deviation()[0]

        # manual steps without qlearning test
        Q.reset(seed=s)
        print(f"seed #{s} just evolution:")
        for _ in range(25):
            Q._env.step()
            print(Q._env.mean_and_deviation(), Q._env.population_size, Q._env.sigma)
        no_q=Q._env.mean_and_deviation()[0]

        c+=1 if with_q<no_q else 0
    print(f"Q learning helped in {c}/50 cases")


if __name__=="__main__":
    functions_to_learn = [f28, f7, f3, f1, f5, f27, f9, f21, f25, f23]
    functions_to_test = [f4]
    actions_P = [-3, -1, 0, 1, 3]
    actions_M = [-0.1, -0.05, 0, 0.05, 0.1]
    p_size = 30
    Q = learn(functions_to_learn, actions_P, actions_M, p_size)
    while True:
        f = eval(input("nazwa funkcji, np. f1: "))
        test(Q, f)