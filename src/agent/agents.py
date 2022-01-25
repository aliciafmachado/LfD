"""
Mapping from model names to models.
"""

from src.agent.ac.ac import Policy
from src.agent.baselines.bc import MLP

mapping_models = {'ac': Policy, 'bc': MLP}