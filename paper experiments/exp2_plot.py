from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()
df = pd.read_pickle("exp2_df")


x = df.explode('Spearman').reset_index()
grouped_df = df.groupby(["Algorithm","dim","Effective dim", "Samples"], as_index=False)
df_mean = grouped_df.mean()#.groupby('Seed').mean()

plotdf = df_mean.pivot(index=["dim","Algorithm"], columns='Samples', values='Spearman')
sns.heatmap(plotdf, cmap="YlGnBu")
plt.show()