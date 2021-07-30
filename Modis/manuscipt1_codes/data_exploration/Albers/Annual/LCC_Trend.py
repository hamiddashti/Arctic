from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm

def linreg(X, Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    """
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det

in_dir = ("/data/home/hamiddashti/nasa_above/outputs/data_analyses/Annual/"
          "Geographics/Figures_MS1/")
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

years = np.arange(1985, 2013)

data = []
for year in years:
    a = pd.read_csv(in_dir + "confusion_table_normalized_" + str(year) +
                    ".csv")
    a = a.drop(a.columns[0], axis=1)
    b = a.values
    data.append(b)
data = np.array(data)

Y = data
X = np.arange(len(Y))


def bootstrap(Y, X, n, seed):
    np.random.seed(seed)
    slopes_bootstrap = []
    for j in range(0, n):
        sample_index = np.random.choice(range(0, len(Y)), len(Y))
        X_samples = X[sample_index]
        y_samples = Y[sample_index, :, :]
        s, intecepts = np.apply_along_axis(linreg, 0, y_samples, X_samples)
        slopes_bootstrap.append(s)
    slopes = np.array(slopes_bootstrap)
    slopes[:, 0, 0].mean()
    slopes_mean = np.mean(slopes, axis=0)
    slopes_std = np.std(slopes, axis=0)
    return np.round(slopes_mean, 5), np.round(slopes_std, 5)


slope_mean, intercept_mean = bootstrap(Y, X, 10000, 1)
slope_mean = slope_mean[0:7, 0:7]
intercept_mean = intercept_mean[0:7, 0:7]

initial_state = ["EF", "DF", "Shrub", "Herb", "Sparse", "Barren", "Fen"]
final_state = ["EF", "DF", "Shrub", "Herb", "Sparse", "Barren", "Fen"]

plt.close()
ar = 1.0  # initial aspect ratio for first trial
wi = 13  # width of the whole figure in inches, ...
hi = wi * ar
rows, cols = 1, 1
gs = gridspec.GridSpec(rows, cols)
fig = plt.figure(figsize=(wi, hi))
for k in range(0, rows * cols):
    ax = plt.subplot(gs[k])
    ax.set_title("Trend in LCC over ABR", fontsize=18, fontweight="bold")
    im = ax.imshow(slope_mean, cmap="bwr")
    ax.set_xticks(np.arange(len(final_state)))
    ax.set_yticks(np.arange(len(initial_state)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(final_state, fontsize=16)
    ax.set_yticklabels(initial_state, fontsize=16)
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(initial_state)):
        for j in range(len(final_state)):
            # if i > j:
            #     continue
            text = ax.text(j,
                           i,
                           str(slope_mean[i, j]) + "\n(\u00B1" +
                           str(intercept_mean[i, j]) + ")",
                           ha="center",
                           va="center",
                           color="black",
                           weight="heavy",
                           fontsize=14)

plt.draw()
xmin, xmax = ax.get_xbound()
ymin, ymax = ax.get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * rows / cols
fig.set_figheight(wi * y2x_ratio)
gs.tight_layout(fig)
plt.savefig(out_dir + "LCC_Trend.png")
print("all done!")