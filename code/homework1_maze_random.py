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
Maze Agent Random
'''

import gym
import gym_maze
import time

MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 500
SLEEP_TIME = 0.1
env = gym.make("maze-homework1-5x5-v0")
env.reset()
env.render()

for episode in range(MAX_EPISODES):
  obs = env.reset()
  total_reward = 0

  print("\nEpisode %d running..." % (episode+1))

  for step in range(MAX_STEPS_PER_EPISODE):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    total_reward += reward
    
    time.sleep(SLEEP_TIME)

    env.render()

    if env.is_game_over():
      exit()

    if done:
      print("Finished after %d steps with total reward = %f." % (step+1, total_reward))
      break

    if step == MAX_STEPS_PER_EPISODE - 1:
      print("Finished without solving after max number of steps", MAX_STEPS_PER_EPISODE)