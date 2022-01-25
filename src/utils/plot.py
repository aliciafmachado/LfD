"""
Calls plot_utils to do the plots.
"""

import argparse

parser = argparse.ArgumentParser(description='Behavioural Cloning')
parser.add_argument('--data_path', type=str, default='demos.pickle')
parser.add_argument('--task', type=str, default='plot_trajectories')
args = parser.parse_args()

# TODO: call functions from plot_utils.py