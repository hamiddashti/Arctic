import os
import xarray as xr
import landsat_functions
import imp
import numpy as np
import pandas as pd
import time

initial_time = time.time()

# -------------------------------------------------------------------------
# The only inputs to the script
# ------------------------------------------------------------------------

in_dir = "/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/"  # The input directory for LC files
out_dir = "/data/home/hamiddashti/mnt/nasa_above/working/landsat/time_series/LCC_outputs/"  # The output directory where tables and figures will be saved
file_name = "lulc_h08v03.tif"  # LC file name


# -------------------------------------------------------------------------
#                          Setting up the scene
# --------------------------------------------------------------------------

print("\n ##### Setting up the scene ########")

start_time = time.time()

lulc_file = in_dir + file_name
lulc = xr.open_rasterio(lulc_file, chunks={"x": 5000, "y": 5000})
lulc.load()  # For the sake of Dask

crs = lulc.rio.crs
lon_number = lulc.sizes["x"]
lat_number = lulc.sizes["y"]
number_of_pixels = lon_number * lat_number
number_of_classes = np.arange(1, 11)
class_nemes = [
    "Evergreen Forest",
    "Deciduous Forest",
    "Shrubland",
    "Herbaceous",
    "Sparsely Vegetated",
    "Barren",
    "Fen",
    "Bog",
    "Shallows/Littoral",
    "Water",
]

print(
    "\n----Setting up the scene is over.\
 total elappsed time: %s minutes ---"
    % (np.around((time.time() - start_time), 2) / 60)
)


# --------------------------------------------------------------------------
# Extracting the temporal changes of classes
# --------------------------------------------------------------------------

print("\n###### Extracting the temporal changes of classes ##########")
print("\nThere is no rolling over time")
start_time = time.time()

change_table = landsat_functions.change_LC(lulc, number_of_pixels)
landsat_functions.my_plot_lulc_percent(
    change_table, "Percent Cover", out_dir + "class_percent_without_roll.png"
)

# Extract the percent of the area that classes were not change between 1984-2014
lulc_std = lulc.std(
    "band"
)  # if variance  of class number is zero then no chages in that class

No_change_index = lulc_std.where(lulc_std == 0)

No_change_percent = (No_change_index.count() / number_of_pixels) * 100
# write outputs
with open(out_dir + "Stats.txt", "w") as text_file:
    print(
        "------ Stats for class changes with no rolling window -----\n", file=text_file
    )
    print(
        f"Number of classes is {len(number_of_classes)} which includes:\n",
        file=text_file,
    )
    print(str(class_nemes).strip("[ ]"), file=text_file)
    print("\nTable of changes without any rolling: \n", file=text_file)
    print(change_table, file=text_file)
    print(
        f"\nIn total: {np.round(No_change_percent.values,2)} of the region has not been changed (with no rolling)",
        file=text_file,
    )
    print("-----------------------------------------------------", file=text_file)


print(
    "\nExtracting the temporal changes of classes is done.\
 total elappsed time: %s minutes ---"
    % (np.around((time.time() - start_time), 2) / 60)
)
print("\nThe bar chart figure and table is saved in the output folder ")
print("\n--------Table of changes without any rolling: \n")
print(change_table)


# -------------------------------------------------------------------
# Getting the mode to filter out random class chages in time dimension
# best guides for this:
# https://stackoverflow.com/questions/56329034/is-there-a-way-to-aggregate-an-xarrray-dataarray-by-calculating-the-mode-for-eac
# https://stackoverflow.com/questions/48510784/xarray-rolling-mean-with-weights/48512802#48512802
# https://stackoverflow.com/questions/50520835/xarray-simple-weighted-rolling-mean-example-using-construct
# -------------------------------------------------------------------

print("\n############ Calculate the mode of classes along time ############ ")
start_time = time.time()

roll_obj = lulc.rolling(band=3)
roll_obj_construct = roll_obj.construct("windows")
lulc_rolled = landsat_functions.mode(roll_obj_construct, dim="windows")
print(
    "\n ---- Mode is calculated in %s minutes ---"
    % (np.around((time.time() - start_time), 2) / 60)
)
print("\n ---- saving the LULC_rolled netcdf file")
lulc_rolled.to_netcdf("LULC_08_03_rolled.nc")


# --------------------------------------------------------------------------
# Extracting the temporal changes of rolled classes
# --------------------------------------------------------------------------
print("\n###### Extracting the temporal changes of classes ##########")
print("\n There is no rolling over time")
start_time = time.time()

change_table_rolled = landsat_functions.change_LC(lulc_rolled, number_of_pixels)
landsat_functions.my_plot_lulc_percent(
    change_table_rolled, "Percent Cover", out_dir + "class_percent_rolled.png"
)

# Extract the percent of the area that classes were not change between 1984-2014
lulc_rolled_std = lulc_rolled.std(
    "band"
)  # if std  of class number is zero then no chages in that class

No_change_index = lulc_rolled_std.where(lulc_rolled_std == 0)
No_change_map = lulc_rolled.where(lulc_rolled_std == 0).isel(
    band=0
)  # Map of pixels where there is no change for three decades
No_change_map.rio.set_crs(crs)
No_change_map.rio.to_raster(out_dir + "No_change_map.tif")

No_change_percent = (No_change_index.count() / number_of_pixels) * 100

# write outputs
with open(out_dir + "Stats.txt", "a") as text_file:
    print(
        "\n\n------ Stats for class changes with rolling window (window size = 3) -----\n",
        file=text_file,
    )
    print("\nTable of changes with rolling: \n", file=text_file)
    print(change_table_rolled, file=text_file)
    print(
        f"\nIn total: {np.around(No_change_percent.values,2)} of the region has not been changed",
        file=text_file,
    )
    print("-----------------------------------------------------", file=text_file)

print(
    "\nExtracting the temporal changes of classes with rolling is done.\
 total elappsed time: %s minutes ---"
    % (np.around((time.time() - start_time), 2) / 60)
)
print("\nThe bar chart figure and table is saved in the output folder ")
print("\nTable of changes without any rolling: \n")
print(change_table_rolled)


# ------------------------------------------------------------------
#  Getting stats on changed pixels
# ------------------------------------------------------------------

print("\n ###### Getting stats on changed pixels ###############")
start_time = time.time()

changed_pixels = lulc_rolled.where(
    lulc_rolled_std != 0
)  # This is all pixels that changed

begining = changed_pixels.isel(band=np.arange(0, 3))  # First three years
end = changed_pixels.isel(band=np.arange(28, 31))  # Last three years

begining_var = begining.std("band")  # var zero means classes are the same
end_var = end.std("band")

A = begining_var.where(begining_var == 0)
B = end_var.where(end_var == 0)

changed_pixels_zero_var = changed_pixels.where(
    A == B
)  # This is all the pixels where consistent classes for three years (begining and end classes might be different)

# Calculate number of pixels that the undergone change but the first and last years are simillar
changed_pixels_zero_var_first_year = changed_pixels_zero_var.isel(band=0)
changed_pixels_zero_var_last_year = changed_pixels_zero_var.isel(band=30)

changed_pixels_similar_classes = changed_pixels.where(
    changed_pixels_zero_var_first_year == changed_pixels_zero_var_last_year
)

changed_pixels_similar_classes_percent = (
    changed_pixels_similar_classes.isel(band=0).count() / number_of_pixels
) * 100

# Calculate number of pixels that the undergone change and the first and last years are different
changed_pixels_different_classes = changed_pixels.where(
    changed_pixels_zero_var_first_year != changed_pixels_zero_var_last_year
)
changed_pixels_different_classes_percent = (
    changed_pixels_different_classes.isel(band=0).count() / number_of_pixels
) * 100

with open(out_dir + "Stats.txt", "a") as text_file:
    print(
        "\n\n------ Stats for changed pixels -----\n", file=text_file,
    )
    print(
        f"\nPercent of pixels that changed but initial and final class are simillar: {np.around(changed_pixels_similar_classes_percent.values,2)}",
        file=text_file,
    )
    print(
        f"\nPercent of pixels that changed and initial and final class are different: {np.around(changed_pixels_different_classes_percent.values,2)}",
        file=text_file,
    )
    print("-----------------------------------------------------", file=text_file)

print(
    f"\nPercent of pixels that changed but initial and final class are simillar: {np.around(changed_pixels_similar_classes_percent.values,2)}"
)
print(
    f"\nPercent of pixels that changed and initial and final class are different: {np.around(changed_pixels_different_classes_percent.values,2)}"
)


# Gettin the confusion matrix between first and last three years class changes
A = changed_pixels_zero_var.isel(band=0)
B = changed_pixels_zero_var.isel(band=30)

A = A.where(A != 0)  # Get rid of zero values
B = B.where(B != 0)

cross_table = pd.crosstab(
    A.to_series(),
    B.to_series(),
    rownames=["Initial classes"],
    colnames=["Current classes"],
    margins=True,
)

for i in np.arange(0, cross_table.shape[1] - 1):
    col = int(cross_table.columns[i])
    cross_table = cross_table.rename(columns={col: class_nemes[col - 1]})
for i in np.arange(0, cross_table.shape[0] - 1):
    ind = int(cross_table.index[i])
    cross_table = cross_table.rename(
        columns={col: class_nemes[col - 1]}, index={ind: class_nemes[ind - 1]}
    )

with open(out_dir + "Stats.txt", "a") as text_file:
    print(
        "\n\n------ Confusion matrix between first and last three years -----\n",
        file=text_file,
    )
    print(cross_table, file=text_file)
    print("-----------------------------------------------------", file=text_file)

print("\nConfusion matrix between first and last three years:\n")
print(cross_table)

print(
    "\n---Getting stats on changed pixels is done.\
 total elappsed time: %s minutes ---"
    % (np.around((time.time() - start_time), 2) / 60)
)


print("\n\n########## Finished! ##############")
final_time = time.time()
total_time = np.around((final_time - initial_time) / 60)
print(f"Total elappsed time: {total_time} minutes")
