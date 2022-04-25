##important for dockerizing later on: https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#installing-using-docker

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from deap import benchmarks
from sklearn.ensemble import RandomForestRegressor

from SALib.sample import saltelli,finite_diff, fast_sampler, latin
from SALib.analyze import morris,sobol, dgsm, fast, delta, rbd_fast
from SALib.util import read_param_file
from SALib.sample.morris import sample
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from tqdm import tqdm

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

def saReport(problem, func):

