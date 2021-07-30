# ----------------------------------------------------------
#  The main goal is to regrid RS products to CLM resolution
# ----------------------------------------------------------

# Import libraries
# -----------------
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
import xesmf as xe
from dask.diagnostics import ProgressBar
import matplotlib.pylab as plt
import itertools
import warnings
# Define functions
#-----------------

# Filter LAI based on QC
#-----------------------
# This should be done before regriding

# Open the files using xarray

# in_dir = '/data/ABOVE/MODIS/APPEEARS_LAI/lai/'
# in_dir = '/groups/davidjpmoore/hamiddashti/nasa_above/regriding_modis_clm/'
# in_dir = "/data/ABOVE/MODIS/LAI_CLM_DOMAIN/"

in_dir = "/data/ABOVE/MODIS/APPEEARS_LAI/lai/"
out_dir = "/data/ABOVE/MODIS/APPEEARS_LAI/lai/processed/"

chunks = ({"time": 10, "lat": 5383, "lon": 4045})
lai_ds = xr.open_dataset(in_dir + 'LAI_500m.nc',chunks=chunks)
lai = lai_ds['Lai_500m']
x=lai
def warn_on_large_chunks(x):
    shapes = list(itertools.product(*x.chunks))
    nbytes = [x.dtype.itemsize * np.prod(shape) for shape in shapes]
    if any(nb > 1e9 for nb in nbytes):
        warnings.warn("Array contains very large chunks")

from dask.distributed import Client, LocalCluster
cluster = LocalCluster()  # Create a local cluster  
client = Client(cluster)


if __name__ == "__main__":
    client = Client()