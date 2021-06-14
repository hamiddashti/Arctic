if __name__ == "__main__":

    import xarray as xr
    import rioxarray
    import numpy as np
    import pandas as pd
    from numpy.lib.stride_tricks import as_strided
    import matplotlib.pyplot as plt
    import dask

    # -----------------------------------------------------------------
    # 		Prameteres and paths
    # -----------------------------------------------------------------
    analyses_mode = "Annual"  # ----> Monthly/Growing/Annual/Seasonal
    # in_dir = "/xdisk/davidjpmoore/hamiddashti/nasa_above_data/Final_data/"
    # out_dir = "/xdisk/davidjpmoore/hamiddashti/nasa_above_outputs/"
    input_dir = "/data/ABOVE/Final_data/"
    output_dir = (
        "/data/home/hamiddashti/nasa_above/outputs/Natural_Variability/"
        "Natural_Variability_Annual_outputs/geographic/")

    nband = 7  # number of LUC classes in the analysis
    win_size = 51  # The size of the search window (e.g. 51*51 pixels or searching within 51 km)
    EndPoint = True
    if EndPoint:
        years = (2003, 2013)
    else:
        years = range(2003, 2015)  # 2003-2015

    # ---------------------------------------------------------------------
    #                      Functions used in the script
    # ---------------------------------------------------------------------
    def check_finite(x):
        # This fuction checks if there is any finite values in an array
        # nan means that there are changes in the LULC
        import numpy as np

        if np.isfinite(x).any():
            # return nan if there is nan (it has been changed)
            return True
        else:
            # return 1 if there is no nan which means no change in LULC
            return False

    def no_change(xrd, dim):
        # This function uses the check_finite and highlights the pixels where pixels
        # LULC changed.
        return xr.apply_ufunc(
            check_finite,
            xrd,
            input_core_dims=[[dim]],
            dask="allowed",
            vectorize=True,
        )

    def dist_matrix(x_size, y_size):
        a1 = np.floor(x_size / 2)
        a2 = np.floor(y_size / 2)
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell = (a1, a2)
        dists = np.sqrt((x_arr - cell[0])**2 + (y_arr - cell[1])**2)
        dists[int(a1), int(a2)] = 0
        return dists

    def window_view(data, win_size, type):
        # This is for creating moving windows
        import numpy as np
        from numpy.lib.stride_tricks import as_strided

        win_size = win_size
        win_size_half = int(np.floor(win_size / 2))

        # pad with nan to get correct window for the edges
        if type == "OTHER":
            data = np.pad(
                data,
                (win_size_half, win_size_half),
                "constant",
                constant_values=(np.nan),
            )
            sub_shape = (win_size, win_size)
            view_shape = tuple(np.subtract(data.shape, sub_shape) +
                               1) + sub_shape
            data_view = as_strided(data, view_shape, data.strides * 2)

        elif type == "LULC":
            nband = 7  # number of classes
            data = np.pad(
                data,
                (
                    (0, 0),
                    (win_size_half, win_size_half),
                    (win_size_half, win_size_half),
                ),
                "constant",
                constant_values=np.nan,
            )
            sub_shape = (nband, win_size, win_size)
            view_shape = tuple(np.subtract(data.shape, sub_shape) +
                               1) + sub_shape
            data_view = as_strided(data, view_shape, data.strides * 2)
            # luc_val_not_changed_view = luc_val_not_changed_view.reshape((-1,) + sub_shape)
            data_view = data_view.squeeze()

        return data_view

    def calculate_nv(LST_MEAN,
                     ALBEDO,
                     ET,
                     LUC,
                     nband,
                     years,
                     win_size,
                     dist_m,
                     out_dir,
                     EndPoint=True):

        for k in range(0, len(years) - 1):
            year1 = years[k]
            year2 = years[k + 1]
            print(year2)
            # open LST and LUC tiles

            luc_year1 = LUC.sel(year=year1)
            luc_year2 = LUC.sel(year=year2)

            # luc_year1 = luc_year1.isel(y=range(1495, 1506), x=range(3995, 4006))
            # luc_year2 = luc_year2.isel(y=range(1495, 1506), x=range(3995, 4006))
            # annual_lst = annual_lst.isel(y=range(1495, 1506), x=range(3995, 4006))

            # Taking the difference in LST and LUC
            delta_abs_luc = abs(luc_year2 - luc_year1)
            delta_luc_loss_gain = luc_year2 - luc_year1

            delta_lst_mean_total = LST_MEAN.sel(year=year2) - LST_MEAN.sel(
                year=year1)
            delta_albedo_total = ALBEDO.sel(year=year2) - ALBEDO.sel(
                year=year1)
            delta_et_total = ET.sel(year=year2) - ET.sel(year=year1)
            # In the original LUC dataset, when there is no class present the pixels where assigned 0. To avoid confusion
            # with pixels that that actually there was a class but it hasn't been changed (e.g.luc2006-luc2005 = 0)
            # we set the pixle values that are zero in both years (non existance classes) to nan.
            tmp = xr.ufuncs.isnan(
                delta_abs_luc.where((luc_year1 == 0) & (luc_year2 == 0)))
            # To convert tmp from True/False to one/zero
            mask = tmp.where(tmp == True)
            delta_abs_luc = delta_abs_luc * mask
            delta_luc_loss_gain = delta_luc_loss_gain * mask

            # If any of 7 classes has been changed more than 1 percent we call that a changed pixels
            # so we don't use them to find the natural variability

            changed_pixels = delta_abs_luc.where(delta_abs_luc > 1)

            # Extracting pixels that have been changed
            # True --> changed; False --> not changed
            changed_pixels_mask = no_change(changed_pixels, "band")
            delta_lst_mean_not_changed = delta_lst_mean_total.where(
                changed_pixels_mask == False)
            delta_lst_mean_changed = delta_lst_mean_total.where(
                changed_pixels_mask == True)
            delta_albedo_not_changed = delta_albedo_total.where(
                changed_pixels_mask == False)
            delta_albedo_changed = delta_albedo_total.where(
                changed_pixels_mask == True)
            delta_et_not_changed = delta_et_total.where(
                changed_pixels_mask == False)
            delta_et_changed = delta_et_total.where(
                changed_pixels_mask == True)
            delta_abs_luc_not_changed = delta_abs_luc.where(
                changed_pixels_mask == False)
            delta_abs_luc_changed = delta_abs_luc.where(
                changed_pixels_mask == True)
            delta_luc_loss_gain_changed = delta_luc_loss_gain.where(
                changed_pixels_mask == True)
            delta_luc_loss_gain_not_changed = delta_luc_loss_gain.where(
                changed_pixels_mask == False)
            """ -----------------------------------------------------------------------
                            Extracting the natural variability of LST

            The method is based on the following paper: 
            Alkama, R., Cescatti, A., 2016. Biophysical climate impacts of recent changes
            in global forest cover. Science (80-. ). 351, 600 LP – 604.
            https://doi.org/10.1126/science.aac8083

            * Here we use the concept of numpy stride_trick to create moving windows. 

            !!!!! Be very CAREFUL about using strides as also advised by numpy!!!!! 
            Best way to check it is to constantly checking the shape of arrays and see if 
            they are correct in every step of the work. 
            ------------------------------------------------------------------------"""

            # Stridind up the LST and LUC at changed and not changed areas
            # -------------------------------------------------------------
            lst_mean_val_not_changed = delta_lst_mean_not_changed.values
            lst_mean_val_not_changed_view = window_view(
                lst_mean_val_not_changed, win_size, type="OTHER")
            lst_mean_val_not_changed_view.shape

            lst_mean_val_changed = delta_lst_mean_changed.values
            lst_mean_val_changed_view = window_view(lst_mean_val_changed,
                                                    win_size,
                                                    type="OTHER")
            lst_mean_val_changed_view.shape

            albedo_val_not_changed = delta_albedo_not_changed.values
            albedo_val_not_changed_view = window_view(albedo_val_not_changed,
                                                      win_size,
                                                      type="OTHER")
            albedo_val_not_changed_view.shape

            albedo_val_changed = delta_albedo_changed.values
            albedo_val_changed_view = window_view(albedo_val_changed,
                                                  win_size,
                                                  type="OTHER")
            albedo_val_changed_view.shape

            et_val_not_changed = delta_et_not_changed.values
            et_val_not_changed_view = window_view(et_val_not_changed,
                                                  win_size,
                                                  type="OTHER")
            et_val_not_changed_view.shape

            et_val_changed = delta_et_changed.values
            et_val_changed_view = window_view(et_val_changed,
                                              win_size,
                                              type="OTHER")
            et_val_changed_view.shape

            luc_val_not_changed = delta_abs_luc_not_changed.values
            luc_val_not_changed_view = window_view(luc_val_not_changed,
                                                   win_size,
                                                   type="LULC")
            luc_val_not_changed_view.shape

            luc_val_changed = delta_abs_luc_changed.values
            luc_val_changed_view = window_view(luc_val_changed,
                                               win_size,
                                               type="LULC")
            luc_val_changed_view.shape

            # Calculate the natural variability
            delta_natural_variability_lst_mean = np.empty(
                (lst_mean_val_changed_view.shape[0],
                 lst_mean_val_changed_view.shape[1]))
            delta_natural_variability_lst_mean[:] = np.nan

            delta_natural_variability_albedo = np.empty(
                (albedo_val_changed_view.shape[0],
                 albedo_val_changed_view.shape[1]))
            delta_natural_variability_albedo[:] = np.nan

            delta_natural_variability_et = np.empty(
                (et_val_changed_view.shape[0], et_val_changed_view.shape[1]))
            delta_natural_variability_et[:] = np.nan

            for i in range(0, lst_mean_val_changed_view.shape[0]):
                for j in range(0, lst_mean_val_changed_view.shape[1]):

                    # Each loops goes through each window
                    # Read the lst and LUC value of changed and not changed pixels
                    lst_mean_changed_tmp = lst_mean_val_changed_view[i, j]
                    lst_mean_not_changed_tmp = lst_mean_val_not_changed_view[i,
                                                                             j]
                    albedo_not_changed_tmp = albedo_val_not_changed_view[i, j]
                    et_not_changed_tmp = et_val_not_changed_view[i, j]

                    luc_changed_tmp = luc_val_changed_view[i, j]
                    luc_not_changed_tmp = luc_val_not_changed_view[i, j]

                    # If the center pixel of the window is nan (meaning there is no LULC change in that pixel) skip it
                    win_size_half = int(np.floor(win_size / 2))
                    if np.isnan(lst_mean_changed_tmp[win_size_half,
                                                     win_size_half]):
                        continue

                    # if nan returns False, else returns True: This line tell us what classes exist (True values) in that central pixel
                    center_luc = (np.isfinite(
                        luc_changed_tmp[:, win_size_half,
                                        win_size_half])).reshape(nband, 1, 1)

                    # This is all pixels where classes havent been changed and surrond the target pixel with classes changed
                    other_luc = np.isfinite(luc_not_changed_tmp)
                    mask = (center_luc == other_luc).all(
                        axis=0
                        # True if the center center pixel have exact same classes  as the other classes in unchanged surronding areas
                        # False otherwise
                    )  # This mask is all pixels that have same class as the central pixel

                    lst_mean_not_changed_tmp_masked = np.where(
                        mask == True, lst_mean_not_changed_tmp, np.nan)
                    albedo_not_changed_tmp_masked = np.where(
                        mask == True, albedo_not_changed_tmp, np.nan)
                    et_not_changed_tmp_masked = np.where(
                        mask == True, et_not_changed_tmp, np.nan)
                    dist_mask = np.where(mask == True, dist_m, np.nan)
                    dist_mask[win_size_half, win_size_half] = np.nan
                    weighted_lst_mean = lst_mean_not_changed_tmp_masked / dist_mask
                    weighted_albedo = albedo_not_changed_tmp_masked / dist_mask
                    weighted_et = et_not_changed_tmp_masked / dist_mask

                    delta_natural_variability_lst_mean[i, j] = np.nansum(
                        weighted_lst_mean) / np.nansum(1 / dist_mask)

                    delta_natural_variability_albedo[i, j] = np.nansum(
                        weighted_albedo) / np.nansum(1 / dist_mask)
                    delta_natural_variability_et[
                        i,
                        j] = np.nansum(weighted_et) / np.nansum(1 / dist_mask)
            # Converting a numpy array to xarray dataframe
            delta_lst_mean_nv = delta_lst_mean_total.copy(
                data=delta_natural_variability_lst_mean)

            delta_albedo_nv = delta_albedo_total.copy(
                data=delta_natural_variability_albedo)

            delta_et_nv = delta_et_total.copy(
                data=delta_natural_variability_et)
            # ------------------------------------------------------------
            # Calulating the delta LST casusd by changes in LUC
            delta_lst_mean_lulc = delta_lst_mean_changed - delta_lst_mean_nv
            delta_albedo_lulc = delta_albedo_changed - delta_albedo_nv
            delta_et_lulc = delta_et_changed - delta_et_nv
            # ------------------------------------------------------------

            # Savinng the results
            changed_pixels_mask = changed_pixels_mask.rename({
                "y": "lat",
                "x": "lon"
            })

            changed_pixels_mask.to_netcdf(out_dir + "changed_pixels_mask_" +
                                          str(year2) + ".nc")

            delta_abs_luc_changed = delta_abs_luc_changed.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_abs_luc_changed.to_netcdf(out_dir +
                                            "delta_abs_luc_changed_" +
                                            str(year2) + ".nc")

            delta_abs_luc_not_changed = delta_abs_luc_not_changed.rename({
                "y":
                "lat",
                "x":
                "lon"
            })
            delta_abs_luc_not_changed.to_netcdf(out_dir +
                                                "delta_abs_luc_not_changed_" +
                                                str(year2) + ".nc")

            delta_luc_loss_gain_changed = delta_luc_loss_gain_changed.rename({
                "y":
                "lat",
                "x":
                "lon"
            })
            delta_luc_loss_gain_changed.to_netcdf(
                out_dir + "delta_luc_loss_gain_changed_" + str(year2) + ".nc")

            delta_luc_loss_gain_not_changed = delta_luc_loss_gain_not_changed.rename(
                {
                    "y": "lat",
                    "x": "lon"
                })
            delta_luc_loss_gain_not_changed.to_netcdf(
                out_dir + "delta_luc_loss_gain_not_changed_" + str(year2) +
                ".nc")

            delta_lst_mean_total = delta_lst_mean_total.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_lst_mean_total.to_netcdf(out_dir + "delta_lst_total_" +
                                           str(year2) + ".nc")

            delta_lst_mean_changed = delta_lst_mean_changed.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_lst_mean_changed.to_netcdf(out_dir +
                                             "delta_lst_mean_changed_" +
                                             str(year2) + ".nc")

            delta_lst_mean_not_changed = delta_lst_mean_not_changed.rename({
                "y":
                "lat",
                "x":
                "lon"
            })
            delta_lst_mean_not_changed.to_netcdf(
                out_dir + "delta_lst_mean_not_changed_" + str(year2) + ".nc")

            delta_lst_mean_nv = delta_lst_mean_nv.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_lst_mean_nv.to_netcdf(
                out_dir + "delta_lst_mean_changed_nv_component_" + str(year2) +
                ".nc")

            delta_lst_mean_lulc = delta_lst_mean_lulc.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_lst_mean_lulc.to_netcdf(
                out_dir + "delta_lst_mean_changed_lulc_component_" +
                str(year2) + ".nc")

            delta_albedo_total = delta_albedo_total.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_albedo_total.to_netcdf(out_dir + "delta_albedo_total_" +
                                         str(year2) + ".nc")

            delta_albedo_changed = delta_albedo_changed.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_albedo_changed.to_netcdf(out_dir + "delta_albedo_changed_" +
                                           str(year2) + ".nc")

            delta_albedo_not_changed = delta_albedo_not_changed.rename({
                "y":
                "lat",
                "x":
                "lon"
            })
            delta_albedo_not_changed.to_netcdf(out_dir +
                                               "delta_albedo_not_changed_" +
                                               str(year2) + ".nc")

            delta_albedo_nv = delta_albedo_nv.rename({"y": "lat", "x": "lon"})
            delta_albedo_nv.to_netcdf(out_dir +
                                      "delta_albedo_changed_nv_component_" +
                                      str(year2) + ".nc")

            delta_albedo_lulc = delta_albedo_lulc.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_albedo_lulc.to_netcdf(
                out_dir + "delta_albedo_changed_lulc_component_" + str(year2) +
                ".nc")
            delta_et_total = delta_et_total.rename({"y": "lat", "x": "lon"})
            delta_et_total.to_netcdf(out_dir + "delta_et_total_" + str(year2) +
                                     ".nc")

            delta_et_changed = delta_et_changed.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_et_changed.to_netcdf(out_dir + "delta_et_changed_" +
                                       str(year2) + ".nc")

            delta_et_not_changed = delta_et_not_changed.rename({
                "y": "lat",
                "x": "lon"
            })
            delta_et_not_changed.to_netcdf(out_dir + "delta_et_not_changed_" +
                                           str(year2) + ".nc")

            delta_et_nv = delta_et_nv.rename({"y": "lat", "x": "lon"})
            delta_et_nv.to_netcdf(out_dir + "delta_et_changed_nv_component_" +
                                  str(year2) + ".nc")

            delta_et_lulc = delta_et_lulc.rename({"y": "lat", "x": "lon"})
            delta_et_lulc.to_netcdf(out_dir +
                                    "delta_et_changed_lulc_component_" +
                                    str(year2) + ".nc")

        if EndPoint == False:
            # years = pd.date_range(start=years[1].year.dt.year, end=years[len(years)-1].year.dt.year, freq="A").year
            fname_changed_pixels_mask = []
            for i in range(1, len(years)):
                tmp = out_dir + "changed_pixels_mask_" + str(years[i]) + ".nc"
                fname_changed_pixels_mask.append(tmp)
            changed_pixels_mask_concat = xr.concat(
                [xr.open_dataarray(f) for f in fname_changed_pixels_mask],
                dim=years[1:])
            changed_pixels_mask_concat = changed_pixels_mask_concat.rename(
                {"concat_dim": "year"})
            changed_pixels_mask_concat.to_netcdf(out_dir +
                                                 "changed_pixels_mask.nc")

            fname_delta_lst_total = []
            for i in range(1, len(years)):
                tmp = out_dir + "delta_lst_total_" + str(years[i]) + ".nc"
                fname_delta_lst_total.append(tmp)
            delta_lst_total_concat = xr.concat(
                [xr.open_dataarray(f) for f in fname_delta_lst_total],
                dim=years[1:])
            delta_lst_total_concat = delta_lst_total_concat.rename(
                {"concat_dim": "year"})
            delta_lst_total_concat.to_netcdf(out_dir + "delta_lst_total.nc")

            fname_delta_lst_changed = []
            for i in range(1, len(years)):
                tmp = out_dir + "delta_lst_changed_" + str(years[i]) + ".nc"
                fname_delta_lst_changed.append(tmp)
            delta_lst_changed_concat = xr.concat(
                [xr.open_dataarray(f) for f in fname_delta_lst_changed],
                dim=years[1:])
            delta_lst_changed_concat = delta_lst_changed_concat.rename(
                {"concat_dim": "year"})
            delta_lst_changed_concat.to_netcdf(out_dir +
                                               "delta_lst_changed.nc")

            fname_delta_lst_not_changed = []
            for i in range(1, len(years)):
                tmp = out_dir + "delta_lst_not_changed_" + str(
                    years[i]) + ".nc"
                fname_delta_lst_not_changed.append(tmp)
            delta_lst_not_changed_concat = xr.concat(
                [xr.open_dataarray(f) for f in fname_delta_lst_not_changed],
                dim=years[1:],
            )
            delta_lst_not_changed_concat = delta_lst_not_changed_concat.rename(
                {"concat_dim": "year"})
            delta_lst_not_changed_concat.to_netcdf(out_dir +
                                                   "delta_lst_not_changed.nc")

            fname_delta_lst_changed_nv_component = []
            for i in range(1, len(years)):
                tmp = (out_dir + "delta_lst_changed_nv_component_" +
                       str(years[i]) + ".nc")
                fname_delta_lst_changed_nv_component.append(tmp)
            delta_lst_changed_nv_component_concat = xr.concat(
                [
                    xr.open_dataarray(f)
                    for f in fname_delta_lst_changed_nv_component
                ],
                dim=years[1:],
            )
            delta_lst_changed_nv_component_concat = (
                delta_lst_changed_nv_component_concat.rename(
                    {"concat_dim": "year"}))
            delta_lst_changed_nv_component_concat.to_netcdf(
                out_dir + "delta_lst_changed_nv_component.nc")

            fname_delta_lst_changed_lulc_component = []
            for i in range(1, len(years)):
                tmp = (out_dir + "delta_lst_changed_lulc_component_" +
                       str(years[i]) + ".nc")
                fname_delta_lst_changed_lulc_component.append(tmp)
            delta_lst_changed_lulc_component_concat = xr.concat(
                [
                    xr.open_dataarray(f)
                    for f in fname_delta_lst_changed_lulc_component
                ],
                dim=years[1:],
            )
            delta_lst_changed_lulc_component_concat = (
                delta_lst_changed_lulc_component_concat.rename(
                    {"concat_dim": "year"}))
            delta_lst_changed_lulc_component_concat.to_netcdf(
                out_dir + "delta_lst_changed_lulc_component.nc")

            fname_delta_abs_luc_changed = []
            for i in range(1, len(years)):
                tmp = out_dir + "delta_abs_luc_changed_" + str(
                    years[i]) + ".nc"
                fname_delta_abs_luc_changed.append(tmp)
            delta_abs_luc_changed_concat = xr.concat(
                [xr.open_dataarray(f) for f in fname_delta_abs_luc_changed],
                dim=years[1:],
            )
            delta_abs_luc_changed_concat = delta_abs_luc_changed_concat.rename(
                {"concat_dim": "year"})
            delta_abs_luc_changed_concat.to_netcdf(out_dir +
                                                   "delta_abs_luc_changed.nc")

            fname_delta_abs_luc_not_changed = []
            for i in range(1, len(years)):
                tmp = out_dir + "delta_abs_luc_not_changed_" + str(
                    years[i]) + ".nc"
                fname_delta_abs_luc_not_changed.append(tmp)
            delta_abs_luc_not_changed_concat = xr.concat(
                [
                    xr.open_dataarray(f)
                    for f in fname_delta_abs_luc_not_changed
                ],
                dim=years[1:],
            )
            delta_abs_luc_not_changed_concat = delta_abs_luc_not_changed_concat.rename(
                {"concat_dim": "year"})
            delta_abs_luc_not_changed_concat.to_netcdf(
                out_dir + "delta_abs_luc_not_changed.nc")

            fname_delta_luc_loss_gain_changed = []
            for i in range(1, len(years)):
                tmp = out_dir + "delta_luc_loss_gain_changed_" + str(
                    years[i]) + ".nc"
                fname_delta_luc_loss_gain_changed.append(tmp)
            delta_luc_loss_gain_changed_concat = xr.concat(
                [
                    xr.open_dataarray(f)
                    for f in fname_delta_luc_loss_gain_changed
                ],
                dim=years[1:],
            )
            delta_luc_loss_gain_changed_concat = (
                delta_luc_loss_gain_changed_concat.rename(
                    {"concat_dim": "year"}))
            delta_luc_loss_gain_changed_concat.to_netcdf(
                out_dir + "delta_luc_loss_gain_changed.nc")

            fname_delta_luc_loss_gain_not_changed = []
            for i in range(1, len(years)):
                tmp = (out_dir + "delta_luc_loss_gain_not_changed_" +
                       str(years[i]) + ".nc")
                fname_delta_luc_loss_gain_not_changed.append(tmp)
            delta_luc_loss_gain_not_changed_concat = xr.concat(
                [
                    xr.open_dataarray(f)
                    for f in fname_delta_luc_loss_gain_not_changed
                ],
                dim=years[1:],
            )
            delta_luc_loss_gain_not_changed_concat = (
                delta_luc_loss_gain_not_changed_concat.rename(
                    {"concat_dim": "year"}))
            delta_luc_loss_gain_not_changed_concat.to_netcdf(
                out_dir + "delta_luc_loss_gain_not_changed.nc")

    if analyses_mode == "Monthly":
        # ------------------------------------------------------------------
        # 				Calculate the delta LST and LUC Monthly
        # ------------------------------------------------------------------
        print(
            "Calculating monthly natural variablity and LUC compontents of \u0394LST"
        )
        LUC = xr.open_dataarray(input_dir + "LUC/LULC_2003_2014.nc")
        Months = [
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
        in_dir = input_dir + "Monthly_Mean/"
        # Matrix of distance of each pixel from the central pixel used in inverse
        dist_m = dist_matrix(win_size, win_size)

        def monthly_cal(k, input_dir, out_dir, LUC, Months, nband, years,
                        win_size, dist_m):
            # for k in list(range(0,12)):
            print(f"Month ----> {Months[k]}")
            out_dir = (
                output_dir +
                "/Natural_Variability/Natural_Variability_Monthly_outputs/" +
                Months[k] + "/")
            LST = xr.open_dataarray(input_dir +
                                    "LST_Final/LST/Monthly_Mean/LST_Mean_" +
                                    Months[k] + ".nc")
            LST = LST.rename({"lat": "y", "lon": "x"})
            LST = LST - 273.15
            LST = LST.rename({"time": "year"})
            LST = LST.assign_coords({"year": LST.year.dt.year.values})
            calculate_nv(LST, LUC, nband, years, win_size, dist_m, out_dir)

        dask.config.set(scheduler="processes")
        lazy_results = []
        for k in np.arange(0, 12):
            lazy_result = dask.delayed(monthly_cal)(k, in_dir, out_dir,
                                                    LUC_dir, Months, nband,
                                                    years, win_size, dist_m)
            lazy_results.append(lazy_result)

        from dask.diagnostics import ProgressBar

        with ProgressBar():
            futures = dask.persist(*lazy_results)
            results = dask.compute(*futures)

        print("The monthly results are saved in /data/home/hamiddashti/mnt/"
              "nasa_above/working/modis_analyses/outputs/Natural_Variability/"
              "Natural_Variability_Monthly_outputs/")
    elif analyses_mode == "Growing":
        # ------------------------------------------------------------------
        # 				Calculate the delta LST and LUC Growing
        # ------------------------------------------------------------------
        print("Calculating growing season (Apr-Nov) natural variablity and "
              "LUC compontents of \u0394LST")
        LUC = xr.open_dataarray(input_dir + "LUC/LULC_2003_2014.nc")
        in_dir = input_dir + "Growing_Mean/"
        out_dir = out_dir + "/Natural_Variability/Natural_Variability_Growing_outputs/"
        # annual_lst = xr.open_dataarray(config_paths['annual_lst_path'])
        LST = xr.open_dataarray(in_dir + "lst_mean_growing.nc")
        LST = LST.rename({"lat": "y", "lon": "x"})
        LST = LST - 273.15
        # Matrix of distance of each pixel from the central pixel used in inverse
        dist_m = dist_matrix(win_size, win_size)
        calculate_nv(LST, ALBEDO, ET, LUC, nband, years, win_size, dist_m,
                     out_dir)
        print("The growing season results are saved in" + out_dir)

    elif analyses_mode == "Annual":
        # ------------------------------------------------------------------
        # 				Calculate the delta LST and LUC Annual
        # ------------------------------------------------------------------
        print("Analyses is in Annual mode\n")
        out_dir = (
            "/data/home/hamiddashti/mnt/nasa_above/working/modis_analyses"
            "/outputs/Natural_Variability/Natural_Variability_Annual_outputs/"
            "EndPoints/")
        # annual_lst = xr.open_dataarray(config_paths['annual_lst_path'])
        LUC = xr.open_dataarray(input_dir + "LUC/LULC_2003_2014.nc")
        LST = xr.open_dataarray(input_dir +
                                "LST_Final/LST/Annual_Mean/lst_mean_annual.nc")
        LST = LST.rename({"lat": "y", "lon": "x"})
        LST = LST
        ALBEDO = xr.open_dataarray(
            input_dir + "ALBEDO_Final/Annual_Albedo/Albedo_annual.nc")
        ET = xr.open_dataarray(input_dir + "ET_Final/Annual_ET/ET_Annual.nc")
        LUC = LUC.isel(y=range(1400, 1600), x=range(4400, 4600))
        LST = LST.isel(y=range(1400, 1600), x=range(4400, 4600))
        ALBEDO = ALBEDO.isel(y=range(1400, 1600), x=range(4400, 4600))
        ET = ET.isel(y=range(1400, 1600), x=range(4400, 4600))
        # Matrix of distance of each pixel from the central pixel used in inverse
        dist_m = dist_matrix(win_size, win_size)
        calculate_nv(LST, ALBEDO, ET, LUC, nband, years, win_size, dist_m,
                     out_dir)
        print("The annual results are saved in" + out_dir)

    elif analyses_mode == "Seasonal":
        print("Calculating seasonal natural variablity and LUC compontents of"
              " \u0394LST")
        LUC = xr.open_dataarray(input_dir + "LUC/LULC_2003_2014.nc")
        Seasons = ["DJF", "MAM", "JJA", "SON"]
        # Seasons = ["DJF"]
        in_dir = input_dir + "LST_Final/LST/Seasonal_Mean/"
        # Matrix of distance of each pixel from the central pixel used in inverse
        dist_m = dist_matrix(win_size, win_size)

        def seasonal_cal(
            k,
            input_dir,
            out_dir,
            LUC,
            Seasons,
            nband,
            years,
            win_size,
            dist_m,
            EndPoint,
        ):
            # for k in list(range(0,12)):
            print(f"Season ----> {Seasons[k]}")
            out_dir = (
                output_dir +
                "/Natural_Variability/Natural_Variability_Seasonal_outputs/" +
                Seasons[k] + "/")
            LST = xr.open_dataarray(in_dir + "LST_Mean_" + Seasons[k] + ".nc")
            LST = LST.rename({"lat": "y", "lon": "x"})
            LST = LST - 273.15
            LST = LST.rename({"time": "year"})
            LST = LST.assign_coords({"year": LST.year.dt.year.values})
            calculate_nv(LST,
                         LUC,
                         nband,
                         years,
                         win_size,
                         dist_m,
                         out_dir,
                         EndPoint=True)

        dask.config.set(scheduler="processes")
        lazy_results = []
        for k in np.arange(0, 4):
            lazy_result = dask.delayed(seasonal_cal)(
                k,
                in_dir,
                output_dir,
                LUC,
                Seasons,
                nband,
                years,
                win_size,
                dist_m,
                EndPoint=True,
            )
            lazy_results.append(lazy_result)

        from dask.diagnostics import ProgressBar

        with ProgressBar():
            futures = dask.persist(*lazy_results)
            results = dask.compute(*futures)

        print("The seasonal results are saved in /data/home/hamiddashti/mnt/"
              "nasa_above/working/modis_analyses/outputs/Natural_Variability/"
              "Natural_Variability_Monthly_outputs/")

    print("All Done!")
    ###################     All Done !    #########################
