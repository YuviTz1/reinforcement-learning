import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import time
import retro
import random
from collections import deque

env=gym.make('CartPole-v1')
env.reset()

num_train_episodes=500

class agent:
    def __init__(self, observation_space_shape, action_space_shape):
        self.action_space_shape=action_space_shape
        self.observation_space_shape=observation_space_shape
        self.replay_memory=deque(maxlen=100_000)
        self.model=self.make_model()
        self.target_model=self.make_model()
        self.epsilon=1
        self.max_epsilon=1
        self.min_epsilon=0.01
        self.decay=0.0001

    def make_model(self):
        learning_rate=0.001

        model=keras.Sequential()
        model.add(keras.layers.Dense(128,input_shape=self.observation_space_shape,activation=tf.nn.relu))
        model.add(keras.layers.Dense(128,activation=tf.nn.relu))
        model.add(keras.layers.Dense(128,activation=tf.nn.relu))
        #model.add(keras.layers.Dense(32, activation=tf.nn.relu))
        model.add(keras.layers.Dense(self.action_space_shape,activation='linear'))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),metrics=['accuracy'])
        return model

    def train(self):
        learning_rate=0.7
        discount_factor=0.99

        min_replay_size=1000
        if len(self.replay_memory)<min_replay_size:
            return

        batch_size=128
        mini_batch=random.sample(self.replay_memory, batch_size)
        current_states=np.array([t[0] for t in mini_batch])
        current_q_list=self.model.predict(current_states)
        new_states=np.array([t[3] for t in mini_batch])
        new_q_list=self.target_model.predict(new_states)

        x=[]
        y=[]
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q=reward+discount_factor*np.max(new_q_list[index])*done
            else:
                max_future_q=reward

            current_q=current_q_list[index]
            current_q[action]=(1-learning_rate)*current_q[action]+learning_rate*max_future_q

            x.append(observation)
            y.append(current_q)
        self.model.fit(np.array(x),np.array(y),batch_size=batch_size,verbose=0,shuffle=True)

    def choose_action(self, observations):
        r=np.random.rand()

        if r<=self.epsilon:
            action=env.action_space.sample()
        else:
            observation=observations.reshape([1,observations.shape[0]])
            predicted_value=self.model.predict(observation).flatten()
            action=np.argmax(predicted_value)
        return action

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def learn(self, num_train_episodes):
        self.update_target_model()

        render=False
        steps_to_update_target_model=0

        for episodes in range(num_train_episodes):
            total_training_rewards=0
            observation=env.reset()
            done=False

            while not done:
                steps_to_update_target_model+=1
                # if render==True:
                #     env.render()
                action=self.choose_action(observation)

                new_observation, reward, done, info=env.step(action)
                self.replay_memory.append([observation,action,reward,new_observation,done])

                if steps_to_update_target_model%7==0 or done:
                    self.train()

                observation=new_observation
                total_training_rewards+=reward

                if done:
                    print("Total training rewards: {} after {} steps  epsilon={}".format(total_training_rewards, episodes,self.epsilon))

                    if steps_to_update_target_model>=100:
                        print("updating target model")
                        self.update_target_model()
                        steps_to_update_target_model=0
                    break

                self.epsilon=max(self.min_epsilon,self.epsilon-self.decay)

            if episodes%50==0:
                self.target_model.save('trained_model ' + str(episodes))
            if render==True:
                render=False
            if episodes%20==0:
                render=True

    def load_model(self, path):
        self.model=keras.models.load_model(path)

    def choose_trained_action(self, observations):
        observation = observations.reshape([1, observations.shape[0]])
        predicted_value = self.model.predict(observation).flatten()
        action = np.argmax(predicted_value)
        return action

    def play_using_trained_model(self, num_episodes, path):
        self.load_model(path)
        for _ in range(num_episodes):
            observation=env.reset()
            done=False
            total_reward=0
            while not done:
                env.render()
                action=self.choose_trained_action(observation)
                new_observation, reward, done, info = env.step(action)
                observation=new_observation
                total_reward+=reward

            print("reward = {}; episode = {}".format(total_reward, _))



def main():
    print(env.observation_space)
    print(env.action_space)
    game_agent=agent(env.observation_space.shape, env.action_space.n)
    game_agent.learn(num_train_episodes)

    # play using trained model
    # game_agent=agent(env.observation_space.shape, env.action_space.n)
    # game_agent.play_using_trained_model(100, "CartPole-v1/trained_model 200")

    # random agent
    # env.reset()
    # while(True):
    #     action=env.action_space.sample()
    #     observation ,reward, done, info = env.step(action)
    #     if done:
    #         env.reset()
    #     env.render()

    env.close()


if __name__=="__main__":
    main()
