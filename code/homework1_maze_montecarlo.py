'''
University of Colorado at Colorado Springs
PhD in Computer Science

Class: CS 4080-5080 - Reinforcement Learning - Fall 2021
Professor: Jugal Kalita
Student: Carlos Eugenio Lopes Pires Xavier Torres
Student ID: 110320128
E-mail: clopespi@uccs.edu
Date: September 20, 2021

Homework 1
Maze agent training with Monte Carlo method
'''

import gym
import gym_maze
import numpy as np
import sys
from time import sleep
from homework1_plot_results import plot
import random


def create_equi_random_policy():
  policy = {}
  policy[0] = {0: 0, 1: 0.5, 2: 0.5, 3: 0}
  policy[1] = {0: 0, 1: 0.33, 2: 0.33, 3: 0.33}
  policy[2] = {0: 0, 1: 0.33, 2: 0.33, 3: 0.33}
  policy[3] = {0: 0, 1: 0.33, 2: 0.33, 3: 0.33}
  policy[4] = {0: 0, 1: 0.5, 2: 0, 3: 0.5}
  policy[5] = {0: 0.33, 1: 0.33, 2: 0.33, 3: 0}
  policy[6] = {0: 0.33, 1: 0, 2: 0.33, 3: 0.33}
  policy[7] = {0: 0.33, 1: 0, 2: 0.33, 3: 0.33}
  policy[8] = {0: 0.33, 1: 0, 2: 0.33, 3: 0.33}
  policy[9] = {0: 0.5, 1: 0, 2: 0, 3: 0.5}
  policy[10] = {0: 0.5, 1: 0.5, 2: 0, 3: 0}
  policy[11] = {0: 0, 1: 0, 2: 0, 3: 0}
  policy[12] = {0: 0, 1: 0, 2: 0, 3: 0}
  policy[13] = {0: 0, 1: 0, 2: 0, 3: 0}
  policy[14] = {0: 0, 1: 0, 2: 0, 3: 0}
  policy[15] = {0: 0.33, 1: 0.33, 2: 0.33, 3: 0}
  policy[16] = {0: 0, 1: 0.33, 2: 0.33, 3: 0.33}
  policy[17] = {0: 0, 1: 0, 2: 0.5, 3: 0.5}
  policy[18] = {0: 0, 1: 0.33, 2: 0.33, 3: 0.33}
  policy[19] = {0: 0, 1: 0.5, 2: 0, 3: 0.5}
  policy[20] = {0: 0.5, 1: 0, 2: 0.5, 3: 0}
  policy[21] = {0: 0.5, 1: 0, 2: 0.5, 3: 0}
  policy[22] = {0: 0, 1: 0, 2: 0, 3: 0}
  policy[23] = {0: 0.5, 1: 0, 2: 0.5, 3: 0}
  policy[24] = {0: 0, 1: 0, 2: 0, 3: 0}
  return policy  


def state_to_cell(state):
    size = MAZE_SIZE[0]
    cell = size * state[1] + state[0]
    return int(cell)


def create_state_action_dictionary(env, policy):
    Q = {}
    for key in policy.keys():
         Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q


def test_policy(env, policy, display=True):
  state = env.reset()
  s = state_to_cell(state)

  for _ in range(len(policy)):
    env.render()
    sleep(SLEEP_TIME)
        
    action = int(np.argmax(list(policy[s].values())))    
    state, _, done, _ = env.step(action)        

    print('State:', s+1, '- Action:', ACTION[action])
    # print("\pi(s_{%d}) = %s \\to" % (s+1, ACTION[action]))

    s = state_to_cell(state)

    if done:
      break
  
  env.render()
  sleep(3)


def run_episode(env, policy, episode_number=0):
     obv = env.reset()
     episode = []
     s = state_to_cell(obv)

     for _ in range(MAX_STEPS_PER_EPISODE):
          timestep = []
          timestep.append(s)
          
          n = random.uniform(0, sum(policy[s].values()))
          top_range = 0
          
          for prob in policy[s].items():
                top_range += prob[1]
                if n < top_range:
                      action = prob[0]
                      break 
          state, reward, done, _ = env.step(action)
          s = state_to_cell(state)
          
          timestep.append(action)
          timestep.append(reward)

          episode.append(timestep)

          if done:            
            break
     
     return episode


def monte_carlo_e_soft(env, episodes=100, gamma=0.99, epsilon=0.01):        
    policy = create_equi_random_policy()  # Create equi-random policy, an empty dictionary to store state action probabilities
    Q = create_state_action_dictionary(env, policy) # Empty dictionary for storing rewards for each state-action pair    
    returns = {}

    plot_rewards = []
    
    for j in range(episodes): # Looping through episodes
        G = 0 # Cumulative reward in G (initialized at 0)
        episode = run_episode(env=env, policy=policy, episode_number=j+1) # Store state, action and value respectively 
        
        print("\rEpisode %d/%d. %.2f%%" % (j+1, episodes, (j+1)/episodes*100), end="")
        sys.stdout.flush()        
        
        for i in reversed(range(0, len(episode))):   
            s_t, a_t, r_t = episode[i] 
            state_action = (s_t, a_t)
            G = gamma * G + r_t # Increment total reward by reward on current timestep
            
            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]   
                    
                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action]) # Average reward across episodes
                
                Q_list = list(map(lambda x: x[1], Q[s_t].items())) # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)
                
                A_star = max_Q
                
                for a in policy[s_t].items(): # Update action probability for s_t in policy
                    sum_state_values = abs(sum(policy[s_t].values()))
                    if sum_state_values == 0:
                      sum_state_values = 1
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (epsilon / sum_state_values)
                    else:
                        policy[s_t][a[0]] = (epsilon / sum_state_values)
        
        # Plot chart of the cumulative rewards (G)
        if SHOW_PLOT:
            plot_rewards.append(G)
            plot(plot_rewards)

    return policy


# Create the environment
env = gym.make("maze-homework1-5x5-v0")    

# Training constants
MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
NUM_STATES = np.prod(MAZE_SIZE, dtype=int)
MAX_STEPS_PER_EPISODE = np.prod(MAZE_SIZE, dtype=int) * 100
SLEEP_TIME = 0.5
ACTION = ["N", "S", "E", "W"] # 0, 1, 2, 3
SHOW_PLOT = True

# Algorithm parameters
NUMBER_OF_EPISODES = 100
GAMMA = 0.99
EPSILON = 0.01

if __name__ == "__main__":

    print('Running algorithm to improve initial random policy...')
    policy = monte_carlo_e_soft(env, episodes=NUMBER_OF_EPISODES, gamma=GAMMA, epsilon=EPSILON)
    
    print('\nFinal Policy', policy)
    
    print('Testing policy...')
    test_policy(env, policy)