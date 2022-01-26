"""
Mapping from model names to models.
"""

from src.agent.actor_critic.ac import Policy
from src.agent.baselines.bc import MLP
from src.agent.baselines.random_agent import RandomAgent

mapping_models = {'ac': Policy, 'bc': MLP, 'random': RandomAgent}