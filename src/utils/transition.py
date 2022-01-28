"""
Transition namedtuple.
"""

from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

TransitionTD = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'n_next_state', 'R_cum', 'gamma_n'))