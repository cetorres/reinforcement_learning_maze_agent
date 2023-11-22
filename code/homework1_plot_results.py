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
Util plot function to show a chart of 
the learning progress during the agent training
'''

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(rewards):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Homework 1 - Maze agent training results')
    plt.title('Homework 1 - Maze agent training results')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards (G)')
    plt.plot(rewards)
    plt.ylim(ymin=-.5)
    plt.text(len(rewards) - 1, rewards[-1], str(rewards[-1]))
    plt.show(block = False)
    plt.pause(.1)
