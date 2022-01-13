import gym
import random
import numpy as np
import time

env = gym.make("Taxi-v3")
env.reset()

num_episodes = 50000
max_steps_per_episode = 100
learning_rate=0.7
discount_rate=0.618

exploration_rate=1
max_exploration_rate=1
min_exploration_rate=0.01
decay_rate=0.01

class Agent():
    def __init__(self, env):
        self.action_size=env.action_space.n
        self.observation_size=env.observation_space.n
        print("Action space: ", self.action_size)
        print("Observation space: ", self.observation_size)

    def initialise_qtable(self):
        self.q_table=np.zeros((self.observation_size,self.action_size))

    def get_action(self):
        action=random.choice(range(self.action_size))
        return action

    def argmax(self, state):            #custom argmax function function to randomly break ties
        top = float("-inf")
        ties = []
        for i in range(len(self.q_table[state])):
            if self.q_table[state][i]>top:
                top,ties=self.q_table[state][i], [i]
            elif self.q_table[state][i]==top:
                ties.append(i)

        index=np.random.choice(ties)
        return index

agent=Agent(env)
agent.initialise_qtable()

#training phase
for _ in range(num_episodes):
    state=env.reset()
    for step in range(max_steps_per_episode):
        r=random.uniform(0,1)
        if(r<exploration_rate):
            action=env.action_space.sample()
        else:
            action=agent.argmax(state)

        new_state, reward, done, info=env.step(action)
        # update qtable according to bellman equation
        agent.q_table[state][action]=agent.q_table[state][action]*(1-learning_rate)+learning_rate*(reward+discount_rate*np.max(agent.q_table[new_state,:]))
        state=new_state
        if done == True:
            break
    exploration_rate=min_exploration_rate+(max_exploration_rate-min_exploration_rate)*np.exp(-decay_rate*_)

#agent playing the game
for episode in range(10):
    state=env.reset()
    done=False
    print("EPISODE ", episode+1, "\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        env.render()
        time.sleep(0.3)
        action=np.argmax(agent.q_table[state][:])
        new_state, reward, done, info = env.step(action)
        if done:
            break
        state=new_state

env.close()