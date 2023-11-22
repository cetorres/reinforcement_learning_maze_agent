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
Generate Maze Data
'''

import numpy as np
import os

# Homework 1 maze representation
maze_homework1_5x5 = [ 
  [3, 3, 1, 3, 3],
  [3, 3, 2, 2, 1],
  [3, 3, 8, 2, 4],
  [3, 3, 8, 2, 3],
  [3, 3, 8, 2, 3]
]

maze_path = 'gym_maze/envs/maze_samples/maze_homework1_5x5.npy'
np.save(maze_path, maze_homework1_5x5)

data_array = np.load(maze_path)

print('Homework 1 Maze')
print(data_array)
print('Saved homework 1 maze at:', maze_path)