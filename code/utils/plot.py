"""
File to create plots and etc.
"""

import seaborn as sns
from matplotlib import pyplot as plt

sns.lineplot(data=flights, x="year", y="passengers")