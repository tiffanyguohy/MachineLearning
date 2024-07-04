import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm 

class QLAgent:

    def __init__(self, env, play_env): 
        self.env = env
        self.play_env = play_env

        #number of action
        a = env.action_space.n

        # number of observation(state)
        o = env.observation_space.n

        print (a, o)

        self.qtable = np.zeros((o,a)) 

    #helper function update Q values on the table
    def learnQ(self, state, action, reward, value):
        oldv = self.qtable[state, action]
        if oldv == 0 :
            self.qtable[state, action] = reward
        else:
            self.qtable[state, action] = oldv + self.alpha * (reward + self.gamma * value - oldv)

    #helper function, find max rewards of the next step
    def learn(self, state1, action1, reward, state2):
        maxQNew = np.max(self.qtable[state2])
        self.learnQ(state1, action1, reward, maxQNew)   
        
    def train(self, neps, alpha, gamma, epsilon):

        # set current state
        cstate, _ = self.env.reset()

        self.alpha = alpha
        self.gamma = gamma
        
        rewards = []

        for i in tqdm(range (neps)):

            #select an random action, and get info
            if np.random.rand()<epsilon:
                action = self.env.action_space.sample() 
            else: 
                action = np.random.choice(np.flatnonzero(self.qtable[cstate] == self.qtable[cstate].max()))
        

            observation, reward, done, info, _ = self.env.step(action)

            # update qtable
            self.learn(cstate, action, reward, observation )

            # update cstate
            cstate = observation    
    
            # array for plotting
            rewards.append(reward)

            #reports rewards
            #if i %100 == 0:
                #print(reward)

            if done:
                cstate, _ = self.env.reset()

        plt.plot(np.arange(i+1),rewards)
        plt.title("Training rewards")
        plt.show

        print(self.qtable)

            

    def play(self, max_iters = 100):
        
        #get initial state 
        cState, _  = self.play_env.reset()

        #render
        rewards = []

        for _ in range(max_iters):

            bestaction = np.random.choice(np.flatnonzero(self.qtable[cState] == self.qtable[cState].max()))

            observation, reward, done, info, _ = self.play_env.step(bestaction)
            
            cState = observation 

            rewards.append(reward)

            #frame = self.play_env.render()
            self.play_env.render()

            #frames.append(frame)

            if done == True:
                break

        print(rewards)
        plt.plot(rewards)
        plt.show()



        
    
    
