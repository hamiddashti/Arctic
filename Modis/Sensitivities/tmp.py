import xarray as xr
import matplotlib.pylab as plt

in_dir = "/data/ABOVE/Final_data/"
out_dir = ("/data/home/hamiddashti/nasa_above/outputs/")

ds = xr.open_dataset("/data/home/hamiddashti/nasa_above/outputs/Sensitivity/"
                     "EndPoints/Annual/all_bands/Confusion_Table2.nc")

nc = ds["NORMALIZED_CONFUSION"]
idx = ds["PIX_INDEX"]
dlcc = ds['DLCC']
dlcc[2, :]
