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
Maze Agent Test
'''

import sys
import numpy as np
import math
import gym
import gym_maze
import time


def start_testing():

    # Render tha maze
    env.render()

    for episode in range(MAX_EPISODES):

        # Reset the environment
        obv = env.reset()

        # Set the initial state
        previous_state = state_to_bucket(obv)
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):

            # Select an action
            action = select_action(previous_state)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward
           
            # Setting up for the next iteration
            previous_state = state

            # Render tha maze
            env.render()
            time.sleep(SIM_SPEED)

            if env.is_game_over():
                sys.exit()                

            if done:
                print("Episode %d finished after %f steps with total reward = %f."
                      % (episode+1, step+1, total_reward))
                break

            elif step >= MAX_STEPS_PER_EPISODE - 1:
                print("Episode %d timed out at %d with total reward = %f." % (episode+1, step+1, total_reward))



def select_action(state):
    action = int(np.argmax(q_table[state]))
    return action


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def load_q_table():
    return np.load('homework1_trained_q_table.npy')


if __name__ == "__main__":

    # Initialize the maze environment
    env = gym.make("maze-homework1-5x5-v0")

    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    # Testing constants
    MAX_EPISODES = 5
    MAX_STEPS_PER_EPISODE = 10
    SIM_SPEED = 0.3

    # Load trained Q table
    q_table = load_q_table()
    
    # Start agent testing
    start_testing()