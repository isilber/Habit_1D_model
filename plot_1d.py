import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib.colors as colors
from os.path import expanduser
from datetime import datetime
from load_1d_model_output import init_load_1d_model_output

"""
-----------------------------------------------------------------------------------
                Israel Silber
            Last update: 7/18/2020
----------------------------------------------------------------------------------
Current methods:
    plot_shaded_prof - plot mean +- SD profiles of multiple parameters.
    plot_shaded_tseries - plot mean =- SD time series of multiple parameters.
    plot_con_mesh - plot field and/or anomalies in colorscale and/or contours.
    init_plt_atts - initialize plotting parameters (incorporates many options).
    deter_num_sub - determine number of panels based on the number of requested
    fields actually existing in the loaded dataset.
"""

#---------------------------------------------------------------------------------------------------------------
def plot_shaded_prof(output_1d, fields_to_plot, time_range=[None, None], Suptitle=None, \
                        YLim=[], AX=None, Fig=None, Xscale='auto', Single_row=True, \
                        no_SD=False, **kwargs):
    """
    ----------------------
    Israel Silber
    Last update: 7/16/2020
    ----------------------
    This method can plot mean +- SD profiles of multiple 1D model output parameters with various
    options and configurations, which also include saving a figure.

    Parameters
    ----------
    output_1d:
        xarray dataset gathering model output.
    fields_to_plot:
        a dictionary containing the output type (as a key) and field to plot (generating
        subplots if multiple fields are requested).
    time_range:
        2 element list providing the time range in fractional hours of simulation time
        to plot (set both to None to plot the full range).
    Suptitle:
        string containing the suptitle (None as default - providing the plotted time
        range).
    YLim:
        2 element list providing y-axis range. If not specified plot the full vertical
        range.
    AX, Fig:
        Axes and Figure objects - should be provided together after making sure that
        the number of requested fields does not exceed the number of subplots in AX
        (otherwise, generating an error), and it is recommended to plot parameters 
        on the same scale (e.g., r_i, r_l) in order to make the figure understandable.
    Xscale:
        string to determine the xscale in the figure. Set to 'linear', 'symlog', or
        'log' to set the scale for all panels, otherwise, scale is set automatically
        based on the metadata in 'init_load_1d_model_output'
    no_SD:
        A boolean; when True, only line plots (mean profiles) are plotted (no shaded
        patch for SD).
    Single_row:
        A boolean: plotting all parameters in a single row of panels with a common y-axis
        if True, plotting following a pre-defined configuration (up to 10 panels; see
        'init_plt_atts') if False.

    Other optional parameters
    -------------------------
    See the 'init_plt_atts' method

    Returns
    -------
    Fig:
        A Figure class object.
    AX:
        An Axes class object (for 1 or more subplots).
    """

    # init configurations
    if not kwargs:
        plt_info = init_plt_atts()
    else:
        plt_info = init_plt_atts(**kwargs)

    num_sub, plot_pairs = deter_num_sub(output_1d, fields_to_plot)
    load_info = init_load_1d_model_output()

    if num_sub == 0:
        print("No match between available and requeted fields to plot - check key and values")
        return  

    # set figure and axes objects    
    if AX is None:
        ax_flag = False
        if Single_row is True:
            Fig, AX = plt.subplots(*[1, num_sub], squeeze=False, \
                dpi=plt_info['dpi'], sharey=True, figsize=plt_info['fig_size']) # use list elements as method input using a splat (*) operator
        else:
            Fig, AX = plt.subplots(*plt_info['sub_sort'][num_sub-1], squeeze=False, \
                dpi=plt_info['dpi'], sharey=True) # use list elements as method input using a splat (*) operator
        Fig.set_visible(plt_info['visible'])
    else:
        ax_flag = True

    # plot panels
    for x in range(num_sub):
        if Single_row is True:
            ax_ind = (0,x)
        else:
            ax_ind = plt_info['sub_ind'][num_sub-1][x] # 2D index for axes
        
        x_variable = np.nanmean(output_1d[plot_pairs[x][0]][plot_pairs[x][1]].sel( \
                                        time=slice(*time_range)), axis=1)
        x_err = np.nanstd(output_1d[plot_pairs[x][0]][plot_pairs[x][1]].sel( \
                                        time=slice(*time_range)), ddof=0, axis=1)
        if plt_info['format'] is "eps": # No transparency in eps format backend.
            if no_SD is False:
                AX[ax_ind].fill_betweenx(output_1d[plot_pairs[x][0]].height, x_variable - x_err, \
                           x_variable + x_err, color=plt_info['color'][0])
                AX[ax_ind].plot(x_variable, output_1d[plot_pairs[x][0]].height, color='black', \
                    linestyle=plt_info['linestyle'], label=plot_pairs[x][1]+",  t = %.1f-%.1f h"\
                    % (tuple(output_1d[plot_pairs[0][0]].time.sel(time=slice(*time_range))[[0, -1]])))
            else:
                AX[ax_ind].plot(x_variable, output_1d[plot_pairs[x][0]].height, color=plt_info['color'][0], \
                    linestyle=plt_info['linestyle'], label=plot_pairs[x][1]+",  t = %.1f-%.1f h"\
                    % (tuple(output_1d[plot_pairs[0][0]].time.sel(time=slice(*time_range))[[0, -1]])))
        else:
            AX[ax_ind].plot(x_variable, output_1d[plot_pairs[x][0]].height, color=plt_info['color'][0], \
                linestyle=plt_info['linestyle'], label=plot_pairs[x][1]+",  t = %.1f-%.1f h"\
                % (tuple(output_1d[plot_pairs[0][0]].time.sel(time=slice(*time_range))[[0, -1]])))
            if no_SD is False:
                AX[ax_ind].fill_betweenx(output_1d[plot_pairs[x][0]].height, x_variable - x_err, \
                           x_variable + x_err, alpha=plt_info['alpha'], color=plt_info['color'][0])
        AX[ax_ind].grid(True)
        if YLim:
            AX[ax_ind].set_ylim(YLim)
        AX[ax_ind].set_xlabel(plot_pairs[x][1] + "   $" + \
                output_1d[plot_pairs[x][0]][plot_pairs[x][1]].units + "$", fontsize=plt_info['fontsize'])
        AX[ax_ind].tick_params(axis='both', which='major', labelsize=plt_info['fontsize'])
        AX[ax_ind].set_title(plt_info['title'], fontsize=plt_info['fontsize'])
        AX[ax_ind].xaxis.get_offset_text().set(size=plt_info['fontsize']) # correct exp size
        AX[ax_ind].yaxis.get_offset_text().set(size=plt_info['fontsize']) # correct exp size
        
        if Xscale in ["linear", "log", "symlog"]:
            AX[ax_ind].set_xscale(Xscale)
        else:
            AX[ax_ind].set_xscale(load_info[plot_pairs[x][0]]["scale"][plot_pairs[x][1]])
        
        if ax_ind[1] == 0:
            AX[ax_ind].set_ylabel(plt_info['ylabel'], fontsize=plt_info['fontsize'])
        
    if Suptitle is None:
        Fig.suptitle("Mean profiles for t = %.2f-%.2f h" \
            % (tuple(output_1d[plot_pairs[0][0]].time.sel( \
                    time=slice(*time_range))[[0, -1]])), fontsize=plt_info['fontsize']+1)
    else:
        Fig.suptitle(Suptitle, fontsize=plt_info['fontsize']+1)
    
    if ax_flag is True: # AX was input
        AX[0,0].legend(fontsize=plt_info['fontsize'])

    # save figure option
    if plt_info['save_fig'] is True:
        plt.savefig(plt_info['fname'], format=plt_info['format'], dpi=plt_info['dpi'])

    return Fig, AX

#---------------------------------------------------------------------------------------------------------------
def plot_shaded_tseries(output_1d, fields_to_plot, hgt_range=["mc","rcld"], Suptitle=None, \
                        XLim=[], AX=None, Fig=None, Yscale='auto', Single_col=True, \
                        no_SD=False, **kwargs):
    """
    ----------------------
    Israel Silber
    Last update: 7/16/2020
    ----------------------
    This method can plot mean +- SD time series of multiple 1D model output parameters with various
    options and configurations, which also include saving a figure.

    Parameters
    ----------
    output_1d:
        xarray dataset gathering model output.
    fields_to_plot:
        a dictionary containing the output type (as a key) and field to plot (generating
        subplots if multiple fields are requested).
    hgt_range:
        2 element list providing the height range in meters to plot.
        provide an output type - field pair existing in the output dataset to plot
        the 3rd quartile height of maximum value of the first requested parameter
        +- 100 m. Otherwise, set both elements in hgt_range to None in order to
    plot the the 3rd quartile height of maximum value of the first requested parameter 
    +- 100 m, i.e., a tendency for higher heights, e.g., to "ignore" low-level
    "artifacts, e.g., useful for rcld).
    Suptitle:
        string containing the suptitle (None as default - providing the plotted height
        range).
    XLim:
        2 element list providing x-axis range. If not specified plot the full temporal
        range.
    AX, Fig:
        Axes and Figure objects - should be provided together after making sure that
        the number of requested fields does not exceed the number of subplots in AX
        (otherwise, generating an error), and it is recommended to plot parameters 
        on the same scale (e.g., r_i, r_l) in order to make the figure understandable.
    Yscale:
        string to determine the yscale in the figure. Set to 'linear', 'symlog', or
        'log' to set the scale for all panels, otherwise, scale is set automatically
        based on the metadata in 'init_load_1d_model_output'
    no_SD:
        A boolean; when True, only line plots (mean time series) are plotted (no shaded
        patch for SD).
    Single_col:
        A boolean: plotting all parameters in a single column of panels with a common x-axis
        if True, plotting following a pre-defined configuration (up to 10 panels; see
        'init_plt_atts') if False. 

    Other optional parameters
    -------------------------
    See the 'init_plt_atts' method

    Returns
    -------
    Fig:
        A Figure class object.
    AX:
        An Axes class object (for 1 or more subplots).
    """

    # init configurations
    if not kwargs:
        plt_info = init_plt_atts()
    else:
        plt_info = init_plt_atts(**kwargs)

    num_sub, plot_pairs = deter_num_sub(output_1d, fields_to_plot)
    load_info = init_load_1d_model_output()

    if num_sub == 0:
        print("No match between available and requeted fields to plot - check key and values")
        return  

    # set figure and axes objects    
    if AX is None:
        ax_flag = False
        if Single_col is True:
            Fig, AX = plt.subplots(*[num_sub,1], squeeze=False, \
                dpi=plt_info['dpi'], sharex=True, figsize=plt_info['fig_size']) # use list elements as method input using a splat (*) operator
        else:
            Fig, AX = plt.subplots(*plt_info['sub_sort'][num_sub-1], squeeze=False, \
                dpi=plt_info['dpi'], sharex=True, figsize=plt_info['fig_size']) # use list elements as method input using a splat (*) operator
        Fig.set_visible(plt_info['visible'])
    else:
        ax_flag = True

    # set height range if not defined or if an output_type-field pair was requested).
    in_data = False
    if hgt_range[0] in output_1d.keys():
        if hgt_range[1] in output_1d[hgt_range[0]]:
            in_data = True
    if in_data is True:
        hgt_range = [output_1d[hgt_range[0]].height.isel( \
            height=output_1d[hgt_range[0]][hgt_range[1]].argmax(dim="height")).quantile( \
            0.75, dim="time").values + vv for vv in [-100, 100]]
    elif hgt_range[0] is None:
        hgt_range = [output_1d[plot_pairs[0][0]].height.isel( \
            height=output_1d[plot_pairs[0][0]][plot_pairs[0][1]].argmax(dim="height")).quantile( \
            0.75, dim="time").values + vv for vv in [-100, 100]]

    # plot panels
    for x in range(num_sub):
        if Single_col is True:
            ax_ind = (x,0)
        else:
            ax_ind = plt_info['sub_ind'][num_sub-1][x] # 2D index for axes
        
        x_variable = np.nanmean(output_1d[plot_pairs[x][0]][plot_pairs[x][1]].sel( \
                                        height=slice(*hgt_range)), axis=0)
        x_err = np.nanstd(output_1d[plot_pairs[x][0]][plot_pairs[x][1]].sel( \
                                        height=slice(*hgt_range)), ddof=0, axis=0)
        if plt_info['format'] is "eps": # No transparency in eps format backend.
            if no_SD is False:
                AX[ax_ind].fill_between(output_1d[plot_pairs[x][0]].time, x_variable - x_err, \
                           x_variable + x_err, color=plt_info['color'][0])
                AX[ax_ind].plot(output_1d[plot_pairs[x][0]].time, x_variable, color='black', \
                    linestyle=plt_info['linestyle'], label=plot_pairs[x][1]+",  h = %.1f-%.1f m"\
                    % (tuple(output_1d[plot_pairs[0][0]].height.sel(height=slice(*hgt_range))[[0, -1]])))
            else:
                AX[ax_ind].plot(output_1d[plot_pairs[x][0]].time, x_variable, color=plt_info['color'][0], \
                    linestyle=plt_info['linestyle'], label=plot_pairs[x][1]+",  h = %.1f-%.1f m"\
                    % (tuple(output_1d[plot_pairs[0][0]].height.sel(height=slice(*hgt_range))[[0, -1]])))
        else:
            AX[ax_ind].plot(output_1d[plot_pairs[x][0]].time, x_variable, color=plt_info['color'][0], \
                linestyle=plt_info['linestyle'], label=plot_pairs[x][1]+",  h = %.1f-%.1f m"\
                % (tuple(output_1d[plot_pairs[0][0]].height.sel(height=slice(*hgt_range))[[0, -1]])))
            if no_SD is False:
                AX[ax_ind].fill_between(output_1d[plot_pairs[x][0]].time, x_variable - x_err, \
                           x_variable + x_err, alpha=plt_info['alpha'], color=plt_info['color'][0])
        AX[ax_ind].grid(True)
        if XLim:
            AX[ax_ind].set_xlim(XLim)
        AX[ax_ind].set_ylabel(plot_pairs[x][1] + "   $" + \
                output_1d[plot_pairs[x][0]][plot_pairs[x][1]].units + "$", fontsize=plt_info['fontsize'])
        AX[ax_ind].tick_params(axis='both', which='major', labelsize=plt_info['fontsize'])
        AX[ax_ind].set_title(plt_info['title'], fontsize=plt_info['fontsize'])
        AX[ax_ind].xaxis.get_offset_text().set(size=plt_info['fontsize']) # correct exp size
        AX[ax_ind].yaxis.get_offset_text().set(size=plt_info['fontsize']) # correct exp size
        
        if Yscale in ["linear", "log", "symlog"]:
            AX[ax_ind].set_yscale(Yscale)
        else:
            AX[ax_ind].set_yscale(load_info[plot_pairs[x][0]]["scale"][plot_pairs[x][1]])
        
        if (Single_col is True and ax_ind[0] == num_sub-1) or \
                (Single_col is False and ax_ind[0] == plt_info['sub_ind'][num_sub-1][x][0]):
            AX[ax_ind].set_xlabel(plt_info['xlabel'], fontsize=plt_info['fontsize'])
        
    if Suptitle is None:
        Fig.suptitle("Mean profiles for h = %.2f-%.2f m" \
            % (tuple(output_1d[plot_pairs[0][0]].height.sel( \
                    height=slice(*hgt_range))[[0, -1]])), fontsize=plt_info['fontsize']+1)
    else:
        Fig.suptitle(Suptitle, fontsize=plt_info['fontsize']+1)
    
    if ax_flag is True: # AX was input
        AX[0,0].legend(fontsize=plt_info['fontsize'])

    # save figure option
    if plt_info['save_fig'] is True:
        plt.savefig(plt_info['fname'], format=plt_info['format'], dpi=plt_info['dpi'])

    return Fig, AX


#---------------------------------------------------------------------------------------------------------------
def plot_con_mesh(output_1d, mesh_field={}, con_field={}, Suptitle=None, YLim=[], XLim=[], \
                        CLim=[], Levels=[], Zscale='auto', Grid=False, Anomalies=False, \
                        val_and_anom=False, mesh_con_anom=[True,False], **kwargs):
    """
    ----------------------
    Israel Silber
    Last update: 7/18/2020
    ----------------------
    This method creates a time-height plot of a single 1D model output parameters with various
    options and configurations, which also include saving a figure.

    Parameters
    ----------
    output_1d:
        xarray dataset gathering model output.
    mesh_field:
        a dictionary containing the output type (as a key) and field to plot in colorscale
        (using only the first field if multiple fields are requested and exist in the output
        dataset).
    con_field:
        a dictionary containing the output type (as a key) and field to plot in contours
        (using only the first field if multiple fields are requested and exist in the output
        dataset).
    Suptitle:
        string containing the suptitle (None as default - providing the long names of the
        illustrated colorscale and/or contour fields).
    YLim:
        2 element list providing y-axis range. If not specified plot the full vertical
        range.
    XLim:
        2 element list providing x-axis range. If not specified plot the full temporal
        range.
    CLim:
        2 element list providing colorbar value range. If not specified plot the full
        field value range is used (only relevant for field values and not anomalies).
    Levels:
        A scalar or a list with number of contours or specified contour levels,
        respectively (only relevant for field values and not anomalies).
    Zscale:
        string to determine the zscale in the figure. Set to 'linear', or 'log' to set
        the scale for all panels (colorscale and contours), otherwise, scale is set
        automatically based on the metadata in 'init_load_1d_model_output'
    Anomalies:
        A boolean; when True, plotting anomalies (deviations from the time-averaged mean
        at each height).
    val_and_anom:
        A boolean; when True (and Anomalies is True), generating two panels - top for
        field values and bottom for field anomalies.
    mesh_con_anome:
        A 2 boolean element list; when True (and Anomalies is True), illustrating anomalies
        of the colorscale field (1st element) and/or anomalies of contour field (2nd element).
        if all elements are False and Anomalies is True, an error is generated.

    Other optional parameters
    -------------------------
    See the 'init_plt_atts' method

    Returns
    -------
    Fig:
        A Figure class object.
    AX_dict:
        A dictionary containing all Axes class, pcolormesh, contour, and colorbar  objects
        (up to 2 objects for each type).
    """

    if Anomalies is True and not any(mesh_con_anom):
        print("Anomalies were requested but set to 'False' for both mesh and contour")
        return

    # init configurations
    if not kwargs:
        plt_info = init_plt_atts()
    else:
        plt_info = init_plt_atts(**kwargs)
    
    num_sub_mesh, mesh_field = deter_num_sub(output_1d, mesh_field)
    num_sub_con, con_field = deter_num_sub(output_1d, con_field)
    load_info = init_load_1d_model_output()
    
    if num_sub_mesh == 0 and num_sub_con == 0:
        print("No match between available and requeted fields to plot - check key and values")
        return  

    # set figure and axes objects
    AX_dict = {"Axes": [], "pcolor": [], "contour": [], "colorbar": []}
    sup_title_str = ""
    if Anomalies is False:
        Fig, AX = plt.subplots(dpi=plt_info['dpi'], figsize=plt_info['fig_size'])
        AX_dict["Axes"].append(AX)
    elif val_and_anom is False: 
        Fig, AX2 = plt.subplots(dpi=plt_info['dpi'], figsize=plt_info['fig_size'])
        AX_dict["Axes"].append(AX2)
    else:
        Fig, (AX,AX2) = plt.subplots(*[2,1], dpi=plt_info['dpi'], figsize=plt_info['fig_size'])
        AX_dict["Axes"].append(AX)
        AX_dict["Axes"].append(AX2)
    Fig.set_visible(plt_info['visible'])

    if num_sub_mesh > 0:
        sup_title_str += output_1d[mesh_field[0][0]][mesh_field[0][1]].long_name + " - colorscale"
        # plot field values
        if Anomalies is False or all([Anomalies, val_and_anom]):
            # zscale is either logarithmic
            if Zscale is "log" or \
               (Zscale not in ["linear", "log"] and  output_1d[mesh_field[0][0]][mesh_field[0][1]].scale is "log"):
                C = AX.pcolormesh(output_1d[mesh_field[0][0]].time, output_1d[mesh_field[0][0]].height, \
                    output_1d[mesh_field[0][0]][mesh_field[0][1]], cmap=plt_info['cmap'], norm=colors.LogNorm())
            # or linear
            else:
                C = AX.pcolormesh(output_1d[mesh_field[0][0]].time, output_1d[mesh_field[0][0]].height,\
                    output_1d[mesh_field[0][0]][mesh_field[0][1]], cmap=plt_info['cmap'])
            if CLim:
                C.set_clim(vmin=CLim[0], vmax=CLim[1])
            AX_dict["pcolor"].append(C)
        
        # Plot anomalies (if requested)
        if Anomalies is True:
            if mesh_con_anom[0] is True:
                # subtract the mean (at every height) from the data array to get deviations (or anomalies)
                plot_arr = output_1d[mesh_field[0][0]][mesh_field[0][1]] - \
                    np.tile(output_1d[mesh_field[0][0]][mesh_field[0][1]].mean(dim="time",skipna=True), \
                    (output_1d[mesh_field[0][0]].dims["time"], 1)).transpose()
                # for the caxis using 99% percentile of abs() to remove "noise"
                max_val = np.quantile(np.abs(plot_arr), 0.99) 
                C2 = AX2.pcolormesh(output_1d[mesh_field[0][0]].time, output_1d[mesh_field[0][0]].height,\
                        plot_arr, cmap=plt_info['cmap_anom'], vmin=-max_val, vmax=max_val)
            else:
                # log
                if Zscale is "log" or \
                   (Zscale not in ["linear", "log"] and  output_1d[mesh_field[0][0]][mesh_field[0][1]].scale is "log"):
                    C2 = AX2.pcolormesh(output_1d[mesh_field[0][0]].time, output_1d[mesh_field[0][0]].height,\
                        output_1d[mesh_field[0][0]][mesh_field[0][1]], cmap=plt_info['cmap'], norm=colors.LogNorm())
                # or linear
                else:
                    C2 = AX2.pcolormesh(output_1d[mesh_field[0][0]].time, output_1d[mesh_field[0][0]].height,\
                        output_1d[mesh_field[0][0]][mesh_field[0][1]], cmap=plt_info['cmap'])
                if CLim:
                    C2.set_clim(vmin=CLim[0], vmax=CLim[1])
            AX_dict["pcolor"].append(C2)

        # set pcolor parameters in one or two axes.
        for ii in range(len(AX_dict["Axes"])):
            aax = AX_dict["Axes"][ii]
            apc = AX_dict["pcolor"][ii]
            acbar = Fig.colorbar(apc, ax=aax)
            AX_dict["colorbar"].append(acbar)
            acbar.ax.set_ylabel("$" + output_1d[mesh_field[0][0]][mesh_field[0][1]].units + "$", \
                                                                        fontsize=plt_info['fontsize'])
            acbar.ax.xaxis.get_offset_text().set(size=plt_info['fontsize']) # correct exp size
            acbar.ax.yaxis.get_offset_text().set(size=plt_info['fontsize']) # correct exp size
            acbar.ax.tick_params(axis='both', which='major', labelsize=plt_info['fontsize'])

    # plot contours (same for field and anomalies).
    if num_sub_con > 0:
        if num_sub_mesh > 0:
            sup_title_str += "\n"
        sup_title_str += output_1d[con_field[0][0]][con_field[0][1]].long_name + " - contours"

        if 'AX' in locals():
            if any(Levels):
                CS = AX.contour(output_1d[con_field[0][0]].time, output_1d[con_field[0][0]].height, \
                    output_1d[con_field[0][0]][con_field[0][1]], Levels, colors=plt_info['contour_color'])
            elif Zscale is "log" or \
               (Zscale not in ["linear", "log"] and  output_1d[con_field[0][0]][con_field[0][1]].scale is "log"):
                CS = AX.contour(output_1d[con_field[0][0]].time, output_1d[con_field[0][0]].height, \
                    output_1d[con_field[0][0]][con_field[0][1]], locator=ticker.LogLocator(), \
                    colors=plt_info['contour_color'])
            else:
                CS = AX.contour(output_1d[con_field[0][0]].time, output_1d[con_field[0][0]].height, \
                    output_1d[con_field[0][0]][con_field[0][1]], colors=plt_info['contour_color'])
            AX_dict["contour"].append(CS)
        if 'AX2' in locals():
            if mesh_con_anom[1] is True:
                plot_arr = output_1d[con_field[0][0]][con_field[0][1]] - \
                    np.tile(output_1d[con_field[0][0]][con_field[0][1]].mean(dim="time",skipna=True), \
                    (output_1d[con_field[0][0]].dims["time"], 1)).transpose()
                CS2 = AX2.contour(output_1d[con_field[0][0]].time, output_1d[con_field[0][0]].height, \
                    plot_arr, colors=plt_info['contour_color'])
            else:
                if any(Levels):
                    CS2 = AX2.contour(output_1d[con_field[0][0]].time, output_1d[con_field[0][0]].height, \
                        output_1d[con_field[0][0]][con_field[0][1]], Levels, colors=plt_info['contour_color'])
                elif Zscale is "log" or \
                   (Zscale not in ["linear", "log"] and  output_1d[con_field[0][0]][con_field[0][1]].scale is "log"):
                    CS2 = AX2.contour(output_1d[con_field[0][0]].time, output_1d[con_field[0][0]].height, \
                        output_1d[con_field[0][0]][con_field[0][1]], locator=ticker.LogLocator(), \
                        colors=plt_info['contour_color'])
                else:
                    CS2 = AX2.contour(output_1d[con_field[0][0]].time, output_1d[con_field[0][0]].height, \
                        output_1d[con_field[0][0]][con_field[0][1]], colors=plt_info['contour_color'])
            AX_dict["contour"].append(CS2)

        # set contour parameters in one or twu axes.
        for ii in range(len(AX_dict["Axes"])):
            aax = AX_dict["Axes"][ii]
            acon = AX_dict["contour"][ii]
            aax.clabel(acon, inline=1, fontsize=plt_info['contour_fontsize'])

    for ii in range(len(AX_dict["Axes"])):
        aax = AX_dict["Axes"][ii]
        aax.xaxis.get_offset_text().set(size=plt_info['fontsize']) # correct exp size
        aax.yaxis.get_offset_text().set(size=plt_info['fontsize']) # correct exp size
        aax.set_title(plt_info['title'], fontsize=plt_info['fontsize'])
        if YLim:
            aax.set_ylim(YLim)
        if XLim:
            aax.set_xlim(XLim)
        if Grid is True:
            aax.grid(color=plt_info['grid_color'], ls='-', alpha=plt_info['alpha'])
        aax.set_ylabel(plt_info['ylabel'], fontsize=plt_info['fontsize'])
        aax.tick_params(axis='both', which='major', labelsize=plt_info['fontsize'])
    AX_dict["Axes"][-1].set_xlabel(plt_info['xlabel'], fontsize=plt_info['fontsize']) # only required in the lowermost panel

    if Anomalies is True:
        AX_dict["Axes"][-1].set_title(AX_dict["Axes"][-1].get_title() + \
            "Anomalies", fontsize=plt_info['fontsize'])

    if Suptitle is None:
        Fig.suptitle(sup_title_str, fontsize=plt_info['fontsize']+1)
    else:
        Fig.suptitle(Suptitle, fontsize=plt_info['fontsize']+1)

    # save figure option
    if plt_info['save_fig'] is True:
        plt.savefig(plt_info['fname'], format=plt_info['format'], dpi=plt_info['dpi'])

    return Fig, AX_dict


#---------------------------------------------------------------------------------------------------------------
def init_plt_atts(**kwargs):
    """
    ----------------------
    Israel Silber
    Last update: 7/17/2020
    ----------------------
    This method initializes plotting parameters.

    Returns
    -------
    plt_info:
        A dictionary containing the plotting parameters in accordance with the call or
        based on default values.
    """

    plt_info = {}
    # visible
    if 'visible' in kwargs.keys():
        plt_info['visible'] = kwargs['visible']
    else:
        plt_info['visible'] = True
    # colormap
    if 'cmap' in kwargs.keys():
        plt_info['cmap'] = kwargs['cmap']
    else:
        plt_info['cmap'] = cm.get_cmap('viridis')
    # anomaly colormap
    if 'cmap_anom' in kwargs.keys():
        plt_info['cmap_anom'] = kwargs['cmap_anom']
    else:
        plt_info['cmap_anom'] = cm.get_cmap('bwr')
    # line color
    if 'color' in kwargs.keys():
        if isinstance(kwargs['color'], list) == True:
            plt_info['color'] = kwargs['color']
        else:
            plt_info['color'] = [kwargs['color']]
    else:
        plt_info['color'] = ['deepskyblue', 'lime', 'purple', 'deeppink', 'teal', \
                                    'gold', 'silver', 'black', 'crimson', 'mediumorchid']
    # linestype
    if 'linestyle' in kwargs.keys():
        plt_info['linestyle'] = kwargs['linestyle']
    else:
        plt_info['linestyle'] = '-'
    # grid color
    if 'grid_color' in kwargs.keys():
        plt_info['grid_color'] = kwargs['grid_color']
    else:
        plt_info['grid_color'] = 'black'
    # shaded patch alpha (1.0 - opaque)
    if 'alpha' in kwargs.keys():
        plt_info['alpha'] = kwargs['alpha']
    else:
        plt_info['alpha'] = 0.3
    # xlabel
    if 'xlabel' in kwargs.keys():
        plt_info['xlabel'] = kwargs['xlabel']
    else:
        plt_info['xlabel'] = "Time [h]"
    # ylabel
    if 'ylabel' in kwargs.keys():
        plt_info['ylabel'] = kwargs['ylabel']
    else:
        plt_info['ylabel'] = "Height [m]"
    # Title
    if 'title' in kwargs.keys():
        plt_info['title'] = kwargs['title']
    else:
        plt_info['title'] = ""
    # Fontsize
    if 'fontsize' in kwargs.keys():
        plt_info['fontsize'] = kwargs['fontsize']
    else:
        plt_info['fontsize'] = 9
    # contour fontsize
    if 'contour_fontsize' in kwargs.keys():
        plt_info['contour_fontsize'] = kwargs['contour_fontsize']
    else:
        plt_info['contour_fontsize'] = plt_info['fontsize'] - 1
    # contour color
    if 'contour_color' in kwargs.keys():
        plt_info['contour_color'] = kwargs['contour_color']
    else:
        plt_info['contour_color'] = ('black')
    # figure size in inches (multiply by dpi to get pixels)
    if 'fig_size' in kwargs.keys():
        plt_info['fig_size'] = kwargs['fig_size']
    else:
        plt_info['fig_size'] = (6.4,4.8) # matplotlib's default
    # DPI - for figure and file export
    if 'dpi' in kwargs.keys():
        plt_info['dpi'] = kwargs['dpi']
    else:
        plt_info['dpi'] = 200
    # True - save figure, visible off, False - show figure
    if 'save_fig' in kwargs.keys():
        plt_info['save_fig'] = kwargs['save_fig']
    else:
        plt_info['save_fig'] = False
    # Figure export format
    if 'format' in kwargs.keys():
        plt_info['format'] = kwargs['format']
        format_spec = True
    else:
        plt_info['format'] = 'png'
        format_spec = False
    # path and name of file to save (considering conflicts with a specified format)
    if 'fname' in kwargs.keys():
        plt_info['fname'] = kwargs['fname']
        # if format was specified then it dominates over filename format (if specified)
        if format_spec is True and plt_info['fname'][-4:] != "." + plt_info['format']:
            plt_info['fname'] += "." + plt_info['format']
            # format was specified in filename (assuming that the last 3 chars denote a legit format).
        elif format_spec is False and plt_info['fname'][-4] == ".":
            plt_info['format'] = plt_info['fname'][-3:]
            # format was not specified in filename (which was specified) or format, so using default format.
        elif format_spec is False and plt_info['fname'][-4] == ".":
            plt_info['fname'] += "." + plt_info['format']        
    else:
        plt_info['fname'] = expanduser("~") + '/model_1d_' + \
            datetime.now().strftime('%Y%m%d.T%H%M%S') + '.' + plt_info['format']
    # subplot arrangement for different number of parameters
    plt_info['sub_sort'] = [[1,1], [1,2], [1,3], [2,2], [2,3], [2,3], [2,4], [2,4], [3,3], [2,5]]
    plt_info['sub_ind'] = [[(0,0)], \
                           [(0,0), (0,1)], \
                           [(0,0), (0,1), (0,2)], \
                           [(0,0), (0,1), (1,0), (1,1)], \
                           [(0,0), (0,1), (0,2), (1,0), (1,1)], \
                           [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)], \
                           [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2)], \
                           [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)], \
                           [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)], \
                           [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4)]]

    return plt_info

#---------------------------------------------------------------------------------------------------------------

def deter_num_sub(output_1d, fields_to_plot):
    """
    ----------------------
    Israel Silber
    Last update: 7/15/2020
    ----------------------
    This method determines the number of subplots required based on the number of requested fields
    actually existing in the model output data structure (both input objects are dictionaries with
    each key consisting of some fields, both of which need to be matched between the two objects).
    
    Parameters
    ----------
    output_1d:
        xarray dataset gathering model output.
    fields_to_plot:
        a dictionary containing the output type (as a key) and field to plot (generating
        subplots if multiple fields are requested).

    Returns
    -------
    num_sub:
        Scalar representing the number of matching fields (number of subplots to be set).
    plot_pairs:
        Dictionary containing pairs of keys (output type) and values (field name) of 
        matching fields.
    """
    num_sub = 0
    plot_pairs = []
    for nn in fields_to_plot.keys():
        if nn in output_1d.keys():
            for ff in output_1d[nn].keys():
                if ff in fields_to_plot[nn]:
                    num_sub += 1
                    plot_pairs.append([nn, ff])
                    
    return num_sub, plot_pairs
