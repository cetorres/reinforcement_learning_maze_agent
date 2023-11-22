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
Maze Agent Train
'''

import sys
import numpy as np
import math
import random
import gym
import gym_maze
from homework1_plot_results import plot
import time

    
def start_training():

    num_consecutive_solvings = 0
    games = 0
    plot_rewards = []

    # Render tha maze
    if RENDER_MAZE:
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

            # Update the Q based on the result
            best_q = np.amax(q_table[state]) 
            q_table[previous_state + (action,)] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * (best_q) - q_table[previous_state + (action,)])

            # Setting up for the next iteration
            previous_state = state

            # Print debug data
            print("\nEpisode: %d" % (episode+1))
            print("Steps: %d" % (step+1))
            print("Action: %d" % action)
            print("State: %s" % str(state))
            print("Reward: %f" % reward)
            print("Best Q: %f" % best_q)
            print("Total reward: %f" % total_reward)
            print("Consecutive solvings: %d" % num_consecutive_solvings)
            print("")

            # Render tha maze
            if RENDER_MAZE:
                env.render()
                time.sleep(SIM_SPEED)

            if env.is_game_over():
                sys.exit()                

            if done:
                print("Episode %d finished after %f steps with total reward = %f (consecutive solvings %d)."
                      % (episode+1, step+1, total_reward, num_consecutive_solvings))

                # Update plot
                games += 1
                if SHOW_PLOT:
                    plot_rewards.append(total_reward)
                    plot(plot_rewards)

                if step <= SOLVED_STEPS:
                    num_consecutive_solvings += 1
                else:
                    num_consecutive_solvings = 0

                break

            elif step >= MAX_STEPS_PER_EPISODE - 1:
                print("Episode %d timed out at %d with total reward = %f." % (episode+1, step+1, total_reward))

        # The best policy is considered achieved when solved over SOLVINGS_TO_END times consecutively
        if num_consecutive_solvings > SOLVINGS_TO_END:
            # Save the trained Q table
            save_q_table()
            break


def select_action(state):
    # Select a random action
    if random.random() < EXPLORE_RATE:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
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


def save_q_table():
    print('\nQ-table')
    print(q_table)
    np.save('homework1_trained_q_table.npy', q_table)


if __name__ == "__main__":
    # Initialize the maze environment
    env = gym.make("maze-homework1-5x5-v0")

    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    # Learning related constants
    EXPLORE_RATE = 0.001  # epsilon -> good to have a function to decrease the value after each episode
    LEARNING_RATE = 0.2
    DISCOUNT_FACTOR = 0.99

    # Training constants
    MAX_EPISODES = 5000
    MAX_STEPS_PER_EPISODE = np.prod(MAZE_SIZE, dtype=int) * 100
    SOLVINGS_TO_END = 40
    SOLVED_STEPS = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 2
    RENDER_MAZE = True
    SHOW_PLOT = True
    SIM_SPEED = 0.008

    # Create the Q-Table to store the state-action pairs
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    # Start training
    start_training()