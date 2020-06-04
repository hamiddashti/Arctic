import xarray as xr
import rioxarray
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
import seaborn as sns
import pandas as pd
from importlib import reload

np.warnings.filterwarnings("ignore")

from scipy import stats
import modis_functions
######################################################################
in_dir = "F:\\MYD21A2\\outputs\\DeltaLST\\Natural_vs_LULC_LST\\Annual_2003_2013\\"
out_dir = "F:\\MYD21A2\\outputs\\DeltaLST\\Natural_vs_LULC_LST\\Annual_2003_2013\\Analyses\\"

class_nemes = [
    "Evergreen Forest",
    "Deciduous Forest",
    "Shrubland",
    "Herbaceous",
    "Sparsely Vegetated/Barren",
    "Wetlands"
]
year1 = 2003
year2 = 2013

# ---------------- Annual LST changes in relation to LST ----------------------
# Plot the cooling effects
delta_annual_lst = xr.open_dataarray(in_dir+'delta_lst_changed_lulc_component.nc')
delta_luc_changed = xr.open_dataarray(in_dir+'delta_abs_luc_changed.nc')


# --------- just to make them align for plotting
delta_annual_lst.rio.set_crs(4326)
delta_luc_changed.rio.set_crs(4326)
delta_luc_changed = delta_luc_changed.rio.reproject_match(delta_annual_lst)

# -------- scatter plot for different classes
print("Graphing the annual LST_LUC")

#luc = delta_luc_changed.isel(band=7)
#outname = out_dir + "AnnualLST_vs_LUC.png"
outname = out_dir + "Outliers_Not_Corrected.png"
reload(modis_functions)
df = modis_functions.lst_luc_plot(
    dluc = delta_luc_changed,
    dvar = delta_annual_lst,
    class_names = class_nemes,
    outname = outname,
    mode="presentation",
    label = "$\Delta$ $LST_{LULC}$($^{\circ}$C)",
    title="Not Corrected for outliers",
    fsize=24,
    multiband = True,
    legend = True
)

with open(out_dir + "Annual_LST_vs_LUC_Report.txt", "w") as text_file:
    print(
        "\nEstimated stats for each class in the linear\
fit to the associated figure:\n",
        file=text_file,
    )
    print(df, file=text_file)
text_file.close()

print(f"##### End of the annual analysis for tile: ##########")
# -------------------- End of the Annual analysis ---------------------------------


# -------------------- Growing season LST changes in relation to LST --------------
growing_lst = xr.open_dataarray(in_dir + "LST\\lst_mean_growing.nc")
delta_growing_lst = growing_lst.sel(year=year2) - growing_lst.sel(year=year1)

# plotting cooling/warming
modis_functions.meshplot(
    delta_growing_lst,
    outname=out_dir + "delta_growing.png",
    mode="presentation",
    label="$\Delta$ LST ($^{\circ}$C)",
    title="Growing season (Apr-Oct) cooling/warming",
)


# just to make lst/luc align for plotting
delta_growing_lst.rio.set_crs(4326)
delta_luc.rio.set_crs(4326)
delta_luc = delta_luc.rio.reproject_match(delta_growing_lst)

# scatter plot for different classes
print("Graphing the grwoing season LST_LUC")
outname = out_dir + "GrowingLST_vs_LUC.png"
df = modis_functions.lst_luc_plot(
    delta_luc,
    delta_growing_lst,
    class_nemes,
    outname,
    mode="presentation",
    title="",
    fsize=18,
    multiband = True,
    legend = True
)


with open(out_dir + "Growing_LST_vs_LUC_Report.txt", "w") as text_file:
    print(
        "\nEstimated stats for each class in the linear\
fit to the associated figure:\n",
        file=text_file,
    )
    print(df, file=text_file)
text_file.close()
# ---------------------------------------------------------------------------------

# ---------------------- Seasonal -------------------------------------------------

season_lst = xr.open_dataarray(
    in_dir + "LST\\lst_mean_season_resample.nc", chunks={"y": 272, "x": 731}
)

delta_spring_lst = season_lst.sel(time=str(year2)).isel(time=0) - season_lst.sel(
    time=str(year1)
).isel(time=0)
delta_spring_lst = delta_spring_lst.to_dataset(name="Spring")
delta_summer_lst = season_lst.sel(time=str(year2)).isel(time=1) - season_lst.sel(
    time=str(year1)
).isel(time=1)
delta_summer_lst = delta_summer_lst.to_dataset(name="Summer")
delta_fall_lst = season_lst.sel(time=str(year2)).isel(time=2) - season_lst.sel(
    time=str(year1)
).isel(time=2)
delta_fall_lst = delta_fall_lst.to_dataset(name="Fall")
delta_winter_lst = season_lst.sel(time=str(year2)).isel(time=3) - season_lst.sel(
    time=str(year1)
).isel(time=3)
delta_winter_lst = delta_winter_lst.to_dataset(name="Winter")

ds = xr.merge([delta_spring_lst, delta_summer_lst, delta_fall_lst, delta_winter_lst])

seasons = ["Spring", "Summer", "Fall", "Winter"]

# ------ ploting the cooling/warming
for i in np.arange(0, 4):
    outname = out_dir + "delta_" + seasons[i] + ".png"
    modis_functions.meshplot(
        ds[seasons[i]],
        outname=outname,
        mode="presentation",
        label="$\ Delta$ LST ($^{\circ}$C)",
        title=seasons[i] + " cooling/warming",
    )


# ------- Scatter plot
# reload(modis_functions)
for i in np.arange(0, 4):
    # just to make lst/luc align for plotting
    print(f"Graphing season:{seasons[i]}")
    ds[seasons[i]].rio.set_crs(4326)
    delta_luc.rio.set_crs(4326)
    delta_luc = delta_luc.rio.reproject_match(ds[seasons[i]])
    # scatter plot for different classes
    outname = out_dir + seasons[i] + "LST_vs_LUC" + ".png"
    df = modis_functions.lst_luc_plot(
        delta_luc,
        ds[seasons[i]],
        class_nemes,
        outname,
        mode="presentation",
        title=seasons[i],
        fsize=18,
        multiband = True,
        legend = True
    )
    with open(out_dir + "Seasonal_LST_vs_LUC_Report.txt", "a") as text_file:
        print(
            f"\nEstimated stats for each class in the linear\
fit to the associated figure {seasons[i]}:\n",
            file=text_file,
        )
        print(df, file=text_file)
    text_file.close()


# -----------------------------------------------------------------------------------

# ------------------------------ Monthly resample -----------------------------------
monthly_lst = xr.open_dataarray(
    in_dir + "LST\\lst_mean_month_resample.nc", chunks={"y": 272, "x": 731}
)

delta_Jan_lst = monthly_lst.sel(time=str(year2)).isel(time=0) - monthly_lst.sel(
    time=str(year1)
).isel(time=0)
delta_Jan_lst = delta_Jan_lst.to_dataset(name="Jan")
delta_Feb_lst = monthly_lst.sel(time=str(year2)).isel(time=1) - monthly_lst.sel(
    time=str(year1)
).isel(time=1)
delta_Feb_lst = delta_Feb_lst.to_dataset(name="Feb")
delta_Mar_lst = monthly_lst.sel(time=str(year2)).isel(time=2) - monthly_lst.sel(
    time=str(year1)
).isel(time=2)
delta_Mar_lst = delta_Mar_lst.to_dataset(name="Mar")
delta_Apr_lst = monthly_lst.sel(time=str(year2)).isel(time=3) - monthly_lst.sel(
    time=str(year1)
).isel(time=3)
delta_Apr_lst = delta_Apr_lst.to_dataset(name="Apr")
delta_May_lst = monthly_lst.sel(time=str(year2)).isel(time=4) - monthly_lst.sel(
    time=str(year1)
).isel(time=4)
delta_May_lst = delta_May_lst.to_dataset(name="May")
delta_Jun_lst = monthly_lst.sel(time=str(year2)).isel(time=5) - monthly_lst.sel(
    time=str(year1)
).isel(time=5)
delta_Jun_lst = delta_Jun_lst.to_dataset(name="Jun")
delta_Jul_lst = monthly_lst.sel(time=str(year2)).isel(time=6) - monthly_lst.sel(
    time=str(year1)
).isel(time=6)
delta_Jul_lst = delta_Jul_lst.to_dataset(name="Jul")
delta_Aug_lst = monthly_lst.sel(time=str(year2)).isel(time=7) - monthly_lst.sel(
    time=str(year1)
).isel(time=7)
delta_Aug_lst = delta_Aug_lst.to_dataset(name="Aug")
delta_Sep_lst = monthly_lst.sel(time=str(year2)).isel(time=8) - monthly_lst.sel(
    time=str(year1)
).isel(time=8)
delta_Sep_lst = delta_Sep_lst.to_dataset(name="Sep")
delta_Oct_lst = monthly_lst.sel(time=str(year2)).isel(time=9) - monthly_lst.sel(
    time=str(year1)
).isel(time=9)
delta_Oct_lst = delta_Oct_lst.to_dataset(name="Oct")
delta_Nov_lst = monthly_lst.sel(time=str(year2)).isel(time=10) - monthly_lst.sel(
    time=str(year1)
).isel(time=10)
delta_Nov_lst = delta_Nov_lst.to_dataset(name="Nov")
delta_Dec_lst = monthly_lst.sel(time=str(year2)).isel(time=11) - monthly_lst.sel(
    time=str(year1)
).isel(time=11)
delta_Dec_lst = delta_Dec_lst.to_dataset(name="Dec")

ds = xr.merge(
    [
        delta_Jan_lst,
        delta_Feb_lst,
        delta_Mar_lst,
        delta_Apr_lst,
        delta_May_lst,
        delta_Jun_lst,
        delta_Jul_lst,
        delta_Aug_lst,
        delta_Sep_lst,
        delta_Oct_lst,
        delta_Nov_lst,
        delta_Dec_lst,
    ]
)

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# ----------- Meshplot
for i in np.arange(0, 12):
    outname = out_dir + "delta_" + months[i] + ".png"
    modis_functions.meshplot(
        ds[months[i]],
        outname=outname,
        mode="presentation",
        label="$\ Delta$ LST ($^{\circ}$C)",
        title=months[i] + " cooling/warming",
    )

# ------------ Scatter plot
for i in np.arange(0, 12):
    # just to make lst/luc align for plotting
    print(f"Graphing month:{months[i]}")
    ds[months[i]].rio.set_crs(4326)
    delta_luc.rio.set_crs(4326)
    delta_luc = delta_luc.rio.reproject_match(ds[months[i]])
    # scatter plot for different classes
    outname = out_dir + months[i] + "LST_vs_LUC" + ".png"
    df = modis_functions.lst_luc_plot(
        delta_luc,
        ds[months[i]],
        class_nemes,
        outname,
        mode="presentation",
        title=months[i],
        fsize=18,
        multiband = True,
        legend = False
    )
    with open(out_dir + "Monthly_LST_vs_LUC_Report.txt", "a") as text_file:
        print(
            f"\nEstimated stats for each class in the linear\
fit to the associated figure {months[i]}:\n",
            file=text_file,
        )
        print(df, file=text_file)
    text_file.close()
