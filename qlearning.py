#Autor: Dominik Sidorczuk, Tomasz Sroka
import numpy as np
from evolution import EvolutionAlgorithm, Point
from environment_base import Environment

class QLearning_evolution:
    # manages q-tables and runs experiments

    def __init__(self, actions_P,actions_M, objective, state_size:tuple = (5,5), population_size=5, proportional_actions=False, alpha=0.2, gamma=0.5, epsilon=0.2, epsilon_min=0, epsilon_decay=0.99) -> None:
        """
        actions_P - valid actions for population state
        actions_M - valid actions for mutation strength state
        objective - function to minimize
        state_size - shape of state space (std,success_rate), for example (10,5) is 10 bins for std and 5 for success_rate
        population - initial points, None generates random points within bounds
        population_size - size
        proportional_actions - if True, actions multiply current parameters instead of adding to them
        """
        self.psize = population_size # for reseting
        self.objective = objective
        self.reset()

        self.actions_P = actions_P
        self.actions_M = actions_M
        self.actions_count = len(actions_P)*len(actions_M)
        self.proportional = proportional_actions

        # @TODO: make this configurable, for example nonlinear bins
        self.bins_std=np.linspace(0,100000,state_size[0]+1)
        self.bins_success_rate=np.linspace(0,1,state_size[1]+1)
        print("bins:",self.bins_std,self.bins_success_rate)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay=epsilon_decay

        self.currectAction=None


        self.Q = np.zeros((len(self.bins_std)-1,len(self.bins_success_rate)-1, self.actions_count))

    def get_environment(self) -> Environment:
        return self._env

    def update_successes(self, success):
        self.success_history.pop()
        self.success_history = [success]+self.success_history

    def bin_state(self,std,success_rate):
        # digitize returns 0 for values under first bin, and len(bins) when over last
        # so return digitize-1, because deviation and rate can't be negative
        # and put all overflowing values to highest bin to prevent errors
        # (better to set correct bin sizes...)
        std_bin = min(np.digitize(std,self.bins_std)-1,len(self.bins_std)-2)
        rate_bin = min(np.digitize(success_rate,self.bins_success_rate)-1,len(self.bins_success_rate)-2)
        return std_bin,rate_bin

    def get_greedy_action(self,std,rate) -> tuple[int,int]:
        #return np.argmax(self.Q[std,rate]) # greedy
        return np.random.choice(np.flatnonzero(self.Q[std,rate] == self.Q[std,rate].max())) # random tie break

    def get_random_action(self):
        return np.random.choice(self.actions_count) # greedy

    def index_to_action(self,index):
        p_idx = index//len(self.actions_P)
        m_idx = index%len(self.actions_P)
        return p_idx,m_idx

    def action_to_index(self,p,m):
        return p*len(self.actions_P)+m

    def select_action(self,std,rate, learn=True) -> tuple[int,int]:
        # if not learning, always select what qtable says is the best
        if learn and np.random.random()<self.epsilon:
            self.currectAction = self.get_random_action()
        else:
            self.currectAction = self.get_greedy_action(std,rate)
        act = self.index_to_action(self.currectAction)
        #if not learn:
            #print("decision:",self.actions_P[act[0]],self.actions_M[act[1]])
        return self.actions_P[act[0]],self.actions_M[act[1]]

    def do_action(self,action_p,action_m):
        #print(f"actions: ",action_p,action_m)
        if self.proportional:
            self._env.population_size*=action_p
            self._env.sigma*=action_m
        else:
            self._env.population_size+=action_p
            self._env.sigma+=action_m

    def update_Qvalues(self,std,rate,action_p,action_m,reward):
        self.Q[std][rate][self.currectAction] = (1-self.alpha)*self.Q[std][rate][self.currectAction]+self.alpha*(reward*self.gamma+np.argmax(self.Q[std,rate]))

    def reset(self, seed=None):
        if seed:
            np.random.seed(seed)
        # gets brand new state
        self.population = np.array([Point() for _ in range(self.psize)])
        self._env=EvolutionAlgorithm(self.population,self.objective)
        self.success_history = [True]*20 # assumes all successes at start


    def episode(self, steps=25, learn=True):
        # calls step() method of evolution class, and based on state picks actions:
        last_mean = None
        for step in range(steps):
            # read current state
            mean,std = self._env.mean_and_deviation()
            rate = sum(self.success_history)/len(self.success_history)
            idx_std,idx_rate = self.bin_state(std,rate)

            action_p,action_m = self.select_action(idx_std,idx_rate,learn)
            self.do_action(action_p,action_m)

            #print(f"s{step}")
            self._env.step()

            # read new state
            mean,std = self._env.mean_and_deviation()
            #print(mean)
            reward = last_mean-mean if last_mean else 0
            #print(reward)

            if learn:
                self.update_Qvalues(idx_std,idx_rate, action_p,action_m, reward)

            # update state
            if last_mean: self.update_successes((mean-last_mean)<0)
            last_mean=mean

    def fit(self, episodes = 10, steps_per_episode=25):
        for ep in range(episodes):
            self.reset()
            self.episode(steps_per_episode)
            self.epsilon = max(self.epsilon_min,self.epsilon*self.epsilon_decay)
            # debug
            print(ep,self._env.mean_and_deviation())
            print("Q:")
            print(self.Q[:,:,:])
            print("eps:",self.epsilon)
