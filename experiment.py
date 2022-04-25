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

from tqdm import tqdm

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()


def meanAbsoluteError(sens, f, d):
    #calculate average distance (euclidean) to end result per algorithm (/per dim)
    labels = ['Morris','Sobol','Fast', "RDB-Fast", "Delta", "DGSM", "pawn", "Pearson", "RF", "Linear"]
    avg_sens = np.mean(sens, axis=1)
    #std_sens = np.std(sens, axis=1) #std sensitivity over all repetitions
    #for rep in np.arange(sens.shape[1]):
    for i in np.arange(sens.shape[2]):
        s_errors = []
        i_stds = []
        for j in np.arange(sens.shape[3]):
            #end result (highest average sample) (samples, algs, dims)
            end_res = avg_sens[-1,i,j]
            squared_error_j = np.square(sens[:-1,:,i,j] - end_res)# / len(avg_sens[:-1,i,j])
            std_j = sens[:,:-1,i,j]
            s_errors.append(squared_error_j)
            i_stds.append(std_j)

        with open('mse-all.csv', mode='a') as file_:
            file_.write("{},{},{},{},{}".format(labels[i], f, d, np.mean(s_errors), np.mean(i_stds)))
            file_.write("\n")  # Next line.
        #print(f, d, labels[i], np.mean(absolute_errors))

def storeResults(x_samples, sens, conf, filename):
    np.save(f"samples{filename}", x_samples)
    np.save(f"sens{filename}", sens)
    np.save(f"conf{filename}", conf)

def plotSensitivity(x_samples, sens, conf, title="Sensitivity scores", filename=""):
    #print(sens.shape, conf.shape) #3, 10, 5, 2 = sample_sizes, reps, algs, dim

    avg_sens = np.mean(sens, axis=1)
    avg_conf = np.mean(conf, axis=1)
    std_sens = np.std(sens, axis=1)

    #colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown']
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    colors = sns.color_palette('husl', n_colors=avg_sens.shape[2])
    labels = ['Morris','Sobol','Fast', "RDB-Fast", "Delta", "DGSM", "pawn", "Pearson", "RF", "Linear"]
    cols = labels
    rows = ['X{}'.format(row) for row in range(avg_sens.shape[2])]

    """ #figure per X
    fig, axes = plt.subplots(avg_sens.shape[2], avg_sens.shape[1], sharey=True, figsize=[20,3*avg_sens.shape[2]])
    fig.suptitle(title)
    
    for j in np.arange(avg_sens.shape[2]):
        for i in np.arange(avg_sens.shape[1]):
            axes[j,i].fill_between(x_samples, (avg_sens[:,i,j]-std_sens[:,i,j]), (avg_sens[:,i,j]+std_sens[:,i,j]), color=conf_colors[i], alpha=0.2 )
            axes[j,i].fill_between(x_samples, (avg_sens[:,i,j]-avg_conf[:,i,j]), (avg_sens[:,i,j]+avg_conf[:,i,j]), color=colors[i], alpha=0.1 )
            axes[j,i].plot(x_samples,avg_sens[:,i,j],color=colors[i], label = labels[i])
            axes[j,i].set_xticks(x_samples)
            axes[j,i].set_xscale('log', base=2)
            #if i > 0:
            axes[j,i].set_ylim([0.0,1.0])
    """
    fig, axes = plt.subplots(2, int(avg_sens.shape[1]/2), sharey=True, figsize=[20,6])
    fig.suptitle(title)
    
    for j in np.arange(avg_sens.shape[2]):
        for i in np.arange(avg_sens.shape[1]):
            axes[int(i/5),i%5].fill_between(x_samples, (avg_sens[:,i,j]-std_sens[:,i,j]), (avg_sens[:,i,j]+std_sens[:,i,j]), color=colors[j], alpha=0.2 )
            axes[int(i/5),i%5].fill_between(x_samples, (avg_sens[:,i,j]-avg_conf[:,i,j]), (avg_sens[:,i,j]+avg_conf[:,i,j]), color=colors[j], alpha=0.1 )
            axes[int(i/5),i%5].plot(x_samples,avg_sens[:,i,j],color=colors[j], linestyle=LINE_STYLES[j%NUM_STYLES] , label = 'X'+str(j))
            axes[int(i/5),i%5].set_xticks(x_samples)
            axes[int(i/5),i%5].set_xscale('log', base=2)
            axes[int(i/5),i%5].set_ylim([0.0,1.0])
            axes[int(i/5),i%5].set_title(labels[i])

    lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0,0]]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    #for ax, col in zip(axes, cols):
    #    ax.set_title(col)

    #for ax, row in zip(axes[:,0], rows):
    #    ax.set_ylabel(row, rotation=0)

    # finally we invoke the legend (that you probably would like to customize...)

    fig.legend(lines, labels)
    fig.tight_layout()
    plt.tight_layout()
    #plt.xlabel("sample size")
    #plt.ylabel("sensitivity index")
    plt.savefig(f"{filename}.pdf")
    #plt.show()
    plt.clf()
    
def runSensitivityExperiment(dim, f, title, filename):
    fun, opt = bn.instantiate(f, iinstance=1)
    problem = {
    'num_vars': dim,
    'names': ['X'+str(x) for x in range(dim)],
    'bounds': [[-5.0, 5.0]] * dim
    }
    x_samples = [8,16,32,64,128,256,512,1024,2048,4096,8192] #,128,256,512,1024,2048,4096,8192 #,8192,16384 ,
    results = []
    conf_results = []
    
    for sample_size in tqdm(x_samples,position=1, leave=False):
        rep_results = []
        rep_conf_results = []
        for rep in tqdm(np.arange(10),position=2, leave=False):
            np.random.seed(rep)
            alg_results = []
            alg_conf_results = []
            X_morris = sample(problem, N=sample_size, num_levels=4, optimal_trajectories=None)
            z_morris =  np.asarray(list(map(fun, X_morris)))

            res_morris = morris.analyze(problem, X_morris, z_morris,
                                        conf_level=0.95,
                                        print_to_console=False,
                                        num_levels=4,
                                        num_resamples=10,
                                        seed=rep)

            mu_star_fixed = np.asarray(res_morris["mu_star"]) / np.sum(res_morris["mu_star"])
            mu_star_conf_fixed = np.asarray(res_morris["mu_star_conf"]) / np.sum(res_morris["mu_star"])

            alg_results.append( mu_star_fixed)
            alg_conf_results.append( mu_star_conf_fixed)

            #Sobol
            X_sobol = saltelli.sample(problem, N=sample_size, calc_second_order=True)
            z_sobol =  np.asarray(list(map(fun, X_sobol)))
            res_sobol = sobol.analyze(problem, z_sobol, print_to_console=False,seed=rep)
            alg_results.append( np.asarray(res_sobol["S1"]))
            alg_conf_results.append( np.asarray(res_sobol["S1_conf"]))
            

            #Fast
            M = 4
            while ((4 * M)**2 > sample_size):
                M -= 1
            if M > 0:
                X_fast = fast_sampler.sample(problem, N=sample_size, M=M, seed=rep)
                z_fast =  np.asarray(list(map(fun, X_fast)))
                res_fast = fast.analyze(problem, z_fast, print_to_console=False,seed=rep)
                alg_results.append( np.asarray(res_fast["S1"]))
                alg_conf_results.append( np.asarray(res_fast["S1_conf"]))
            else:
                alg_results.append(np.zeros(mu_star_fixed.shape))
                alg_conf_results.append(np.zeros(mu_star_fixed.shape))

            #rbd #delta
            X_latin = latin.sample(problem, N=sample_size)
            z_latin =  np.asarray(list(map(fun, X_latin)))
            res_rbd = rbd_fast.analyze(problem, X_latin, z_latin, print_to_console=False,seed=rep)
            res_delta = delta.analyze(problem, X_latin, z_latin, print_to_console=False,seed=rep)
            alg_results.append( np.asarray(res_rbd["S1"]))
            alg_conf_results.append( np.asarray(res_rbd["S1_conf"]))
            alg_results.append( np.asarray(res_delta["S1"]))
            alg_conf_results.append( np.asarray(res_delta["S1_conf"]))

            #dgsm
            X_dgsm = finite_diff.sample(problem, N=sample_size)
            z_dgsm =  np.asarray(list(map(fun, X_dgsm)))
            res_dgsm = dgsm.analyze(problem, X_dgsm, z_dgsm, print_to_console=False)
            
            dgsm_fixed = np.asarray(res_dgsm["dgsm"]) / np.sum(res_dgsm["dgsm"])
            alg_results.append( dgsm_fixed)
            dgsm_conf_fixed = np.asarray(res_dgsm["dgsm_conf"]) / np.sum(res_dgsm["dgsm"])
            alg_conf_results.append( dgsm_conf_fixed)


            #pawn
            res_pawn = pawn.analyze(problem, X_latin, z_latin, S=10, print_to_console=False,seed=rep)
            pawn_fixed = np.asarray(res_pawn["median"])
            alg_results.append(np.array(pawn_fixed))
            alg_conf_results.append(np.zeros(np.array(pawn_fixed).shape))

            #Pearson Correlation
            prs = []
            for col in range(X_latin.shape[1]):
                pr,_ = pearsonr(X_latin[:,col], z_latin)
                prs.append(pr)
            alg_results.append(np.abs(prs))
            alg_conf_results.append(np.zeros(np.array(prs).shape))

            #Random forest
            forest = RandomForestRegressor(random_state=rep)
            forest.fit(X_latin, z_latin)
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
            rf_fixed = np.asarray(importances) / np.sum(importances)
            rf_conf_fixed = np.asarray(std) / np.sum(importances)
            alg_results.append(rf_fixed)
            alg_conf_results.append(rf_conf_fixed)

            #linear model
            reg = LinearRegression().fit(X_latin, z_latin)
            coefs = reg.coef_
            coefs_fixed = np.abs(np.asarray(coefs)) / np.sum(np.abs(coefs))
            alg_results.append(coefs_fixed)
            alg_conf_results.append(np.zeros(coefs_fixed.shape))


            #combine
            rep_results.append(np.asarray(alg_results))
            rep_conf_results.append(np.asarray(alg_conf_results))
        results.append(np.asarray(rep_results))
        conf_results.append(np.asarray(rep_conf_results))

    storeResults(x_samples, np.asarray(results), np.asarray(conf_results), filename)
    #plotSensitivity(x_samples, np.asarray(results), np.asarray(conf_results), title=title, filename=filename)
    meanAbsoluteError(np.asarray(results), f, dim)

from benchmark import bbobbenchmarks as bn

fIDs = bn.nfreeIDs[:]    # for all fcts

for dim in [2,5,10,20]:
    for f in tqdm(fIDs, position=0):
        runSensitivityExperiment(dim, f, title=f"Average Sensitivity Scores per Sample Size on F{f} D{dim}", filename=f"f{f}-d{dim}") #maybe add repetitions