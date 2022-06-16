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
from joblib import Parallel, delayed
from pandas.util import hash_pandas_object

# Apply the default theme
sns.set_theme()


def storeResults(dim, informative_dim, sample_size, seed, spearman, timeres):
    np.save(f"npy/spearman-{dim}-{informative_dim}-{sample_size}-{seed}", spearman)
    np.save(f"npy/time-{dim}-{informative_dim}-{sample_size}-{seed}", timeres)\
    
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def runExperiment(dim, effective_dim, sample_size, rep):
    seed = rep
    f = testFunction(dim, inf, 0.1, seed)
    df = pd.DataFrame(columns =['Algorithm','dim','Effective dim', 'Samples', 'Seed', 'Spearman', 'Time'])
    fun = f.predict
    ground_truth = f.ground_truth
    problem = {
    'num_vars': dim,
    'names': ['X'+str(x) for x in range(dim)],
    'bounds': [[0.0, 1.0]] * dim
    }
    start_data = { 'dim':dim, 'Effective dim':effective_dim, 'Samples': sample_size, 'Seed': seed, 'ground_truth': ground_truth}
    np.random.seed(seed)

    ground_truth_rank = np.argsort(ground_truth.flatten())[::-1][:effective_dim]

    sample_size_morris = int(sample_size / dim)
    if (sample_size_morris > 0):
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
        mu_star_rank = np.argsort(mu_star_fixed)[::-1][:effective_dim]
        tau, _ = stats.kendalltau(mu_star_rank, ground_truth_rank)
        newrow['Tau'] = tau
        newrow['Samples_real'] = len(X_morris)
        newrow['Prediction'] = res_morris["mu_star"]
        newrow['Algorithm'] = "Morris"
        df = df.append(newrow, ignore_index=True)
    else:
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = 0
        newrow['Time'] = 0
        newrow['Algorithm'] = "Morris"
        newrow['Tau'] = np.nan
        df = df.append(newrow, ignore_index=True)

    #Sobol
    sample_size_sobol = int(sample_size / (dim))
    if (sample_size_sobol > 0):
        X_sobol = saltelli.sample(problem, N=sample_size_sobol, calc_second_order=False)
        z_sobol =  np.asarray(list(map(fun, X_sobol)))
        start_time = time.perf_counter()
        res_sobol = sobol.analyze(problem, z_sobol.flatten(), print_to_console=False,seed=rep,calc_second_order=False)
        end_time = time.perf_counter()
        res, _ = stats.spearmanr(np.asarray(res_sobol["S1"]), ground_truth)
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = res
        newrow['Time'] = end_time - start_time
        newrow['Algorithm'] = "Sobol"
        newrow['Prediction'] = res_sobol["S1"]
        newrow['Samples_real'] = len(X_sobol)
        ranking = np.argsort(np.asarray(res_sobol["S1"]))[::-1][:effective_dim]
        tau, _ = stats.kendalltau(ranking, ground_truth_rank)
        newrow['Tau'] = tau
        df = df.append(newrow, ignore_index=True)
    else:
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = 0
        newrow['Time'] = 0
        newrow['Algorithm'] = "Sobol"
        newrow['Tau'] = np.nan
        df = df.append(newrow, ignore_index=True)
    

    #Fast
    M = 4
    
    sample_size_fast = int(sample_size / dim)
    while ((4 * M)**2 > sample_size_fast):
        M -= 1
    if M > 0:
        X_fast = fast_sampler.sample(problem, N=sample_size_fast, M=M, seed=rep)
        z_fast =  np.asarray(list(map(fun, X_fast))).flatten()
        start_time = time.perf_counter()
        res_fast = fast.analyze(problem, z_fast, print_to_console=False,seed=rep)
        end_time = time.perf_counter()
        res, _ = stats.spearmanr(np.asarray(res_fast["S1"]), ground_truth)
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = res
        newrow['Time'] = end_time - start_time
        newrow['Algorithm'] = "Fast"
        newrow['Prediction'] = res_fast["S1"]
        newrow['Samples_real'] = len(X_fast)
        ranking = np.argsort(np.asarray(res_fast["S1"]))[::-1][:effective_dim]
        tau, _ = stats.kendalltau(ranking, ground_truth_rank)
        newrow['Tau'] = tau
        df = df.append(newrow, ignore_index=True)
    else:
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = 0
        newrow['Time'] = 0
        newrow['Algorithm'] = "Fast"
        newrow['Tau'] = np.nan
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
    newrow['Prediction'] = res_rbd["S1"]
    ranking = np.argsort(np.asarray(res_rbd["S1"]))[::-1][:effective_dim]
    tau, _ = stats.kendalltau(ranking, ground_truth_rank)
    newrow['Tau'] = tau
    df = df.append(newrow, ignore_index=True)
    
    start_time = time.perf_counter()
    res_delta = delta.analyze(problem, X_latin, z_latin, print_to_console=False,seed=rep)
    end_time = time.perf_counter()
    res, _ = stats.spearmanr(np.asarray(res_delta["S1"]), ground_truth)
    newrow = copy.deepcopy(start_data)
    newrow['Spearman'] = res
    newrow['Time'] = end_time - start_time
    newrow['Algorithm'] = "Delta"
    newrow['Prediction'] = res_delta["S1"]
    ranking = np.argsort(np.asarray(res_delta["S1"]))[::-1][:effective_dim]
    tau, _ = stats.kendalltau(ranking, ground_truth_rank)
    newrow['Tau'] = tau
    df = df.append(newrow, ignore_index=True)

    #dgsm
    sample_size_dgsm = int(sample_size / (dim))
    if (sample_size_dgsm > 0):
        X_dgsm = finite_diff.sample(problem, N=sample_size_dgsm)
        z_dgsm =  np.asarray(list(map(fun, X_dgsm))).flatten()
        start_time = time.perf_counter()
        res_dgsm = dgsm.analyze(problem, X_dgsm, z_dgsm, print_to_console=False)
        end_time = time.perf_counter()
        res, _ = stats.spearmanr(np.asarray(res_dgsm["dgsm"]), ground_truth)
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = res
        newrow['Time'] = end_time - start_time
        newrow['Algorithm'] = "DGSM"
        newrow['Samples_real'] = len(X_dgsm)
        newrow['Prediction'] = res_dgsm["dgsm"]
        ranking = np.argsort(np.asarray(res_dgsm["dgsm"]))[::-1][:effective_dim]
        tau, _ = stats.kendalltau(ranking, ground_truth_rank)
        newrow['Tau'] = tau
        df = df.append(newrow, ignore_index=True)
    else:
        newrow = copy.deepcopy(start_data)
        newrow['Spearman'] = 0
        newrow['Time'] = 0
        newrow['Algorithm'] = "DGSM"
        newrow['Prediction'] = []
        newrow['Tau'] = np.nan
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
    newrow['Prediction'] = res_pawn["median"]
    ranking = np.argsort(np.asarray(res_pawn["median"]))[::-1][:effective_dim]
    tau, _ = stats.kendalltau(ranking, ground_truth_rank)
    newrow['Tau'] = tau
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
    newrow['Prediction'] = prs
    ranking = np.argsort(np.abs(prs))[::-1][:effective_dim]
    tau, _ = stats.kendalltau(ranking, ground_truth_rank)
    newrow['Tau'] = tau
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
    newrow['Prediction'] = importances
    ranking = np.argsort(np.asarray(importances))[::-1][:effective_dim]
    tau, _ = stats.kendalltau(ranking, ground_truth_rank)
    newrow['Tau'] = tau
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
    newrow['Prediction'] = coefs
    ranking = np.argsort(np.abs(np.asarray(coefs)))[::-1][:effective_dim]
    tau, _ = stats.kendalltau(ranking, ground_truth_rank)
    newrow['Tau'] = tau
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

x_samples = [128,256,512,1024,2048,4096,8192,16384,32768]#
informative_dims  = {2: [2], 8:[6], 16:[8], 32:[16], 128:[32,64], 1024:[64,256], 8192:[128,4048]}
main_df = pd.DataFrame(columns =['Algorithm','dim','Effective dim', 'Samples', 'Seed', 'Spearman', 'Time'])
for dim in tqdm([2,8,16,32,128,1024,8192], position=1, leave=False): #
    inf_dims = informative_dims[dim]
    for inf in inf_dims:
        for sample_size in tqdm(x_samples, position=2, leave=False):
            #loop = asyncio.get_event_loop()                                              # Have a new event loop
            #looper = asyncio.gather(*[runExperiment(dim, inf, sample_size, i) for i in range(10)])         # Run the loop                      
            #results = loop.run_until_complete(looper)  
            results = Parallel(n_jobs=10)(delayed(runExperiment)(dim, inf, sample_size, i) for i in range(10))
            for r in results:
                #dfr = pd.read_json(r)
                main_df = pd.concat([main_df, r])
    main_df.to_pickle("exp2_maindf")