from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from tqdm import tqdm

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()


def plotSensitivity(f, d):
    filename=f"f{f}-d{d}"
    title=f"Average Sensitivity Scores per Sample Size on F{f} D{d}"
    #print(sens.shape, conf.shape) #3, 10, 5, 2 = sample_sizes, reps, algs, dim

    x_samples = np.load(f"npy/samples{filename}.npy")
    sens = np.load(f"npy/sens{filename}.npy")
    conf = np.load(f"npy/conf{filename}.npy")

    avg_sens = np.mean(sens, axis=1)
    avg_samples = np.mean(x_samples, axis=1)
    print(avg_samples.shape)
    avg_conf = np.mean(conf, axis=1)
    std_sens = np.std(sens, axis=1)

    #colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown']
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    colors = sns.color_palette('husl', n_colors=avg_sens.shape[2])
    labels = ['Morris','Sobol','Fast', "RDB-Fast", "Delta", "DGSM", "Pawn", "Pearson", "RF", "Linear"]
    cols = labels
    rows = ['X{}'.format(row) for row in range(avg_sens.shape[2])]

    fig, axes = plt.subplots(2, 5, sharey=True, figsize=[16,6])
    fig.suptitle(title)
    
    i_correct = 0
    for j in np.arange(avg_sens.shape[2]):
        for i in np.arange(avg_sens.shape[1]):
            ind = i
            axes[int(ind/5),ind%5].fill_between(avg_samples[:,i], (avg_sens[:,i,j]-std_sens[:,i,j]), (avg_sens[:,i,j]+std_sens[:,i,j]), color=colors[j], alpha=0.2 )
            #axes[int(ind/5),ind%5].fill_between(x_samples, (avg_sens[:,i,j]-avg_conf[:,i,j]), (avg_sens[:,i,j]+avg_conf[:,i,j]), color=colors[j], alpha=0.1 )
            axes[int(ind/5),ind%5].plot(avg_samples[:,i],avg_sens[:,i,j],color=colors[j], linestyle=LINE_STYLES[j%NUM_STYLES] , label = 'X'+str(j))
            axes[int(ind/5),ind%5].set_xticks([128,256,512,1024,2048,4096,8192,16384,32768])
            axes[int(ind/5),ind%5].set_xscale('log', base=2)
            axes[int(ind/5),ind%5].set_ylim([0.0,1.0])
            axes[int(ind/5),ind%5].set_title(labels[i])
            if (ind/5 >= 2):
                axes[int(ind/5),ind%5].set_xlabel("samples")
            if (ind%5 == 0):
                axes[int(ind/5),ind%5].set_ylabel("normalized sensitivity")

    lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0,0]]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]


    fig.legend(lines, labels)
    fig.tight_layout()
    plt.tight_layout()
    plt.savefig(f"img/{filename}.png")
    #plt.show()
    plt.clf()
    
#plotSensitivity(2,2)

from benchmark import bbobbenchmarks as bn

fIDs = bn.nfreeIDs[:]    # for all fcts


for dim in [2,4,8,16,32,64]:
    for f in tqdm(fIDs, position=0):
        plotSensitivity(f,dim)