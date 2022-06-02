from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from deap import benchmarks
from sklearn.ensemble import RandomForestRegressor

from SALib.sample import saltelli,finite_diff, fast_sampler, latin
from SALib.analyze import morris,sobol, dgsm, fast, delta, rbd_fast, pawn
from SALib.util import read_param_file
from SALib.sample.morris import sample
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
from tqdm import tqdm
import time
from itertools import groupby
import pandas as pd

# Import seaborn
import seaborn as sns
import copy

# Apply the default theme
sns.set_theme()


def storeResults(dim, informative_dim, sample_size, seed, spearman, timeres):
    np.save(f"npy/spearman-{dim}-{informative_dim}-{sample_size}-{seed}", spearman)
    np.save(f"npy/time-{dim}-{informative_dim}-{sample_size}-{seed}", timeres)\
    
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def runExperiment(dim, effective_dim, fun, sample_size, ground_truth, seed, df):
    problem = {
    'num_vars': dim,
    'names': ['X'+str(x) for x in range(dim)],
    'bounds': [[0.0, 1.0]] * dim
    }
    start_data = { 'dim':dim, 'Effective dim':effective_dim, 'Samples': sample_size, 'Seed': seed}
    np.random.seed(seed)

    X_morris = sample(problem, N=sample_size, num_levels=4, optimal_trajectories=None)
    z_morris =  np.asarray(list(map(fun, X_morris)))
    start_time = time.perf_counter()
    res_morris = morris.analyze(problem, X_morris, z_morris.flatten(),
                                conf_level=0.95,
                                print_to_console=False,
                                num_levels=4,
                                num_resamples=10,
                                seed=rep)
    end_time = time.perf_counter()
    mu_star_fixed = np.asarray(res_morris["mu_star"]) / np.sum(res_morris["mu_star"])
    res, _ = stats.spearmanr(mu_star_fixed, ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "Morris"
    df = df.append(newrow, ignore_index=True)

    #Sobol
    X_sobol = saltelli.sample(problem, N=sample_size, calc_second_order=True)
    z_sobol =  np.asarray(list(map(fun, X_sobol)))
    start_time = time.perf_counter()
    res_sobol = sobol.analyze(problem, z_sobol.flatten(), print_to_console=False,seed=rep)
    end_time = time.perf_counter()
    res, _ = stats.spearmanr(np.asarray(res_sobol["S1"]), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "Sobool"
    df = df.append(newrow, ignore_index=True)
    

    #Fast
    M = 4
    while ((4 * M)**2 > sample_size):
        M -= 1
    if M > 0:
        X_fast = fast_sampler.sample(problem, N=sample_size, M=M, seed=rep)
        z_fast =  np.asarray(list(map(fun, X_fast))).flatten()
        start_time = time.perf_counter()
        res_fast = fast.analyze(problem, z_fast, print_to_console=False,seed=rep)
        end_time = time.perf_counter()
        res, _ = stats.spearmanr(np.asarray(res_fast["S1"]), ground_truth)
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = res
        newrow['Time'] = end_time - start_time
        newrow['Algorithm'] = "Fast"
        df = df.append(newrow, ignore_index=True)
    else:
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = 0
        newrow['Time'] = 0
        newrow['Algorithm'] = "Fast"
        df = df.append(newrow, ignore_index=True)

    #rbd #delta
    X_latin = latin.sample(problem, N=sample_size)
    z_latin =  np.asarray(list(map(fun, X_latin))).flatten()
    start_time = time.perf_counter()
    res_rbd = rbd_fast.analyze(problem, X_latin, z_latin, print_to_console=False,seed=rep)
    end_time = time.perf_counter()
    if (all_equal(np.asarray(res_rbd["S1"]))):
        res = -1
    else:
        res, _ = stats.spearmanr(np.asarray(res_rbd["S1"]), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "RBD-Fast"
    df = df.append(newrow, ignore_index=True)
    
    start_time = time.perf_counter()
    res_delta = delta.analyze(problem, X_latin, z_latin, print_to_console=False,seed=rep)
    end_time = time.perf_counter()
    res, _ = stats.spearmanr(np.asarray(res_delta["S1"]), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "Delta"
    df = df.append(newrow, ignore_index=True)

    #dgsm
    X_dgsm = finite_diff.sample(problem, N=sample_size)
    z_dgsm =  np.asarray(list(map(fun, X_dgsm))).flatten()
    start_time = time.perf_counter()
    res_dgsm = dgsm.analyze(problem, X_dgsm, z_dgsm, print_to_console=False)
    end_time = time.perf_counter()
    res, _ = stats.spearmanr(np.asarray(res_dgsm["vi"]), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "DGSM"
    df = df.append(newrow, ignore_index=True)

    #pawn
    start_time = time.perf_counter()
    res_pawn = pawn.analyze(problem, X_latin, z_latin, S=10, print_to_console=False,seed=rep)
    end_time = time.perf_counter()
    res, _ = stats.spearmanr(np.asarray(res_pawn["median"]), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "PAWN"
    df = df.append(newrow, ignore_index=True)

    #Pearson Correlation
    prs = []
    start_time = time.perf_counter()
    for col in range(X_latin.shape[1]):
        pr,_ = pearsonr(X_latin[:,col], z_latin)
        prs.append(pr)
    end_time = time.perf_counter()
    res, _ = stats.spearmanr(np.abs(prs), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "Pearson"
    df = df.append(newrow, ignore_index=True)

    #Random forest
    start_time = time.perf_counter()
    forest = RandomForestRegressor(random_state=rep)
    forest.fit(X_latin, z_latin)
    importances = forest.feature_importances_
    end_time = time.perf_counter()
    res, _ = stats.spearmanr(np.asarray(importances), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "RF"
    df = df.append(newrow, ignore_index=True)
    
    #linear model
    start_time = time.perf_counter()
    reg = LinearRegression().fit(X_latin, z_latin)
    coefs = reg.coef_
    end_time = time.perf_counter()
    res, _ = stats.spearmanr(np.abs(np.asarray(coefs)), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "Linear"
    df = df.append(newrow, ignore_index=True)

    return df
  
from sklearn.utils import check_random_state

class testFunction():
    def __init__(self, n_features, n_informative, noise=0.1, seed=42):
        # Generate a ground truth model with only n_informative features being non
        # zeros (the other features are not correlated to y and should be ignored
        # by a sparsifying regularizers such as L1 or elastic net)
        self.generator = check_random_state(seed=seed)
        ground_truth = np.zeros((n_features, 1))
        ground_truth[:n_informative, :] = 100 * self.generator.uniform(
            size=(n_informative, 1)
        )
        self.ground_truth = ground_truth
        self.noise = noise

    def predict(self, X):
        y = np.dot(X, self.ground_truth)
        if self.noise > 0.0:
            y += self.generator.normal(scale=self.noise, size=y.shape)
        return y


import warnings
warnings.filterwarnings('ignore')

x_samples = [16,32,64,128,256,512,1024,2048,4096,8192]#
informative_dims  = {2: [2], 5:[4], 10:[6], 20:[10], 100:[16,26], 1000:[42,300], 10000:[68,3000]}
df = pd.DataFrame(columns =['Algorithm','dim','Effective dim', 'Samples', 'Seed', 'Spearman', 'Time'])
for dim in tqdm([2,5,10,20,100,1000,10000], position=1, leave=False): #
    inf_dims = informative_dims[dim]
    for inf in inf_dims:
        for sample_size in tqdm(x_samples, position=2, leave=False):
            for rep in np.arange(10):
                f = testFunction(dim, inf, 0.1)
                df = runExperiment(dim, inf, f.predict, sample_size, f.ground_truth, rep, df)
    df.to_pickle("exp2_df") 