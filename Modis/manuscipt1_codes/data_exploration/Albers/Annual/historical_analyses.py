import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


def linreg(X, Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    """
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x * x
        Syy = Syy + y * y
        Sxy = Sxy + x * y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx) / det, (Sxx * Sy - Sx * Sxy) / det


in_dir = (
    "/data/home/hamiddashti/nasa_above/outputs/historical_change_bio/final/")
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses/test/"

dlst = xr.open_dataarray(in_dir + "dlst_hist.nc")
dlst_mean = dlst.mean(dim=("lat", "lon"))
dalbedo = xr.open_dataarray(in_dir + "dalbedo_hist.nc")
dalbedo_mean = dalbedo.mean(dim=("lat", "lon"))
det = xr.open_dataarray(in_dir + "det_hist.nc")
det_mean = det.mean(dim=("lat", "lon"))

plt.close()
dlst_mean.plot()
plt.savefig(out_dir + "test.png")

plt.close()
dalbedo_mean.plot()
plt.savefig(out_dir + "test_albedo.png")

plt.close()
det_mean.plot()
plt.savefig(out_dir + "test_et.png")

dalbedo_mean
dlst_mean
det_mean

x = dlst_mean.values
a, b = linreg(range(len(x)), x)

x = det_mean.values
a, b = linreg(range(len(x)), x)
