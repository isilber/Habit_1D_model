import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from os.path import expanduser
from load_1d_model_output import load_1d_model_output, init_load_1d_model_output
import plot_1d

# ---------------------- create a directory for the figure files ---------------------------------------
"""
Creating a directory for figures in the home directory.
"""
plot_path=expanduser("~/")+'1D_model_plots/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# ------------------------------- load data ------------------------------------------------------------
load_info = init_load_1d_model_output()
output_1d = load_1d_model_output( \
    run_path=os.path.abspath(os.getcwd()) + "/example_sim/", \
    time_range=[2., 9.], types_to_load=["mc", "tke", "prof"], \
    fields_to_load={"mc": \
    ["zm", "rt", "rvap", "rcld", "rrain", "rpice", "rhoiavg", "cpice", "aspect", "amean"], \
    "prof": ["TKE", "thetail", "htrt", "htrt_lw", "htrt_sw", "w"]})

# ---------------------- plot shaded profiles ----------------------------------------------------------
Fig, AX = plot_1d.plot_shaded_prof(output_1d, \
    {"tke":["TKE", "tke_dissip", "tke_vtrspt", "tke_shear", "tke_buoy", "tke_ttrspt"]}, \
    format='eps', fontsize=6, YLim=[0, 1.7e3], color='slategray', time_range=[2.5,3.5], fig_size=(9.6,5.4))
Fig, AX = plot_1d.plot_shaded_prof(output_1d, \
    {"tke":["TKE", "tke_dissip", "tke_vtrspt", "tke_shear", "tke_buoy", "tke_ttrspt"]}, \
    fontsize=6, YLim=[0, 1.7e3], color='deepskyblue', linestyle='--', time_range=[5,6], fig_size=(9.6,5.4), 
    format='eps', AX=AX, Fig=Fig)
Fig, AX = plot_1d.plot_shaded_prof(output_1d,
    {"tke":["TKE", "tke_dissip", "tke_vtrspt", "tke_shear", "tke_buoy", "tke_ttrspt"]}, \
    fontsize=6, YLim=[0, 1.7e3], color='purple', linestyle='-.', \
    Suptitle="Mean profiles for AWARE simulation", time_range=[7,9], fig_size=(9.6,5.4), AX=AX, Fig=Fig, \
    save_fig=True, format='png', fname=plot_path+'prof_TKE_1d_awr.png')

Fig, AX = plot_1d.plot_shaded_prof(output_1d, \
    {"prof":["TKE", "thetail", "htrt", "w"], "mc":["rt", "rvap", "rcld", "rrain", "rpice", "amean"]}, \
    Xscale="linear", fontsize=6, YLim=[0, 1.7e3], color='black', time_range=[2.5,3.5], fig_size=(9.6,5.4))
Fig, AX = plot_1d.plot_shaded_prof(output_1d, \
    {"prof":["TKE", "thetail", "htrt", "w"], "mc":["rt", "rvap", "rcld", "rrain", "rpice", "amean"]}, \
    Xscale="linear", fontsize=6, YLim=[0, 1.7e3], color='deepskyblue', linestyle='--', time_range=[5,6], \
    fig_size=(9.6,5.4), AX=AX, Fig=Fig)
Fig, AX = plot_1d.plot_shaded_prof(output_1d,
    {"prof":["TKE", "thetail", "htrt", "w"], "mc":["rt", "rvap", "rcld", "rrain", "rpice", "amean"]}, \
    Xscale="linear", fontsize=6, YLim=[0, 1.7e3], color='purple', linestyle='-.', \
    Suptitle="Mean profiles for AWARE simulation", time_range=[7,9], fig_size=(9.6,5.4), AX=AX, Fig=Fig, \
    save_fig=True, format='png', fname=plot_path+'prof_MC_TKE_1d_awr.png')


# ----------------------plot shaded time series---------------------------------------------------------

# focus on the height range where tke often maximizes.
Fig, AX = plot_1d.plot_shaded_tseries(output_1d, \
    {"tke":["TKE", "tke_dissip", "tke_vtrspt", "tke_shear", "tke_buoy", "tke_ttrspt"]}, \
        fontsize=6, XLim=[2., 9.], fig_size=(9.6,8.1), hgt_range=["tke","TKE"], color='teal')
# focus on the 1400-1600 m layer (just because).
Fig, AX = plot_1d.plot_shaded_tseries(output_1d, \
    {"tke":["TKE", "tke_dissip", "tke_vtrspt", "tke_shear", "tke_buoy", "tke_ttrspt"]}, \
    fontsize=6, XLim=[2., 9.], fig_size=(9.6,8.1), hgt_range=[1400, 1600], color='black', \
    linestyle='--', AX=AX, Fig=Fig, save_fig=True, format='png', \
    fname=plot_path+'ts_TKE_1d.png')

# focus on the height range where rcld often maximizes.
Fig, AX = plot_1d.plot_shaded_tseries(output_1d, \
    {"mc":["rt", "rcld", "rrain", "rpice", "amean"], "prof":["TKE", "thetail", "htrt", "w"]}, \
    Yscale="linear", fig_size=(9.6,8.1), fontsize=6, XLim=[2., 9.], hgt_range=["mc","rcld"], color='black')
# focus on the height range where rpice often maximizes.
Fig, AX = plot_1d.plot_shaded_tseries(output_1d, \
    {"mc":["rt", "rcld", "rrain", "rpice", "amean"], "prof":["TKE", "thetail", "htrt", "w"]}, \
    Yscale="linear", fig_size=(9.6,8.1), fontsize=6, XLim=[2., 9.], hgt_range=["mc","rpice"], color='gold', \
    linestyle='--', AX=AX, Fig=Fig)
# focus on the height range where amean often maximizes.
Fig, AX = plot_1d.plot_shaded_tseries(output_1d, \
    {"mc":["rt", "rcld", "rrain", "rpice", "amean"], "prof":["TKE", "thetail", "htrt", "w"]}, \
    Yscale="linear", fig_size=(9.6,8.1), fontsize=6, XLim=[2., 9.], hgt_range=["mc","amean"], color='deeppink', \
    linestyle=':', AX=AX, Fig=Fig,save_fig=True, format='png', \
    fname=plot_path+'ts_MC_TKE_1d_awr.png')

# ----------------------plot time vs. height curtains ---------------------------------------------------------
# Only anomalies
Fig, AX_dict = plot_1d.plot_con_mesh(output_1d, mesh_field={"mc": ["rpice"]}, con_field={"mc": ["rcld"]}, \
     YLim=[0, 1.7e3], Anomalies=True, format='png', save_fig=True, \
                            fname=plot_path+'mesh_anom_MC_1d_awr.png')
# Anomalies and field values.
Fig, AX_dict = plot_1d.plot_con_mesh(output_1d, mesh_field={"prof": ["htrt"]}, con_field={"tke": ["TKE"]}, \
    YLim=[0, 1.7e3], Anomalies=True, val_and_anom=True, mesh_con_anom=[True,True], cmap='PiYG', \
    CLim=[-0.2,0.2], format='png', save_fig=True, fname=plot_path+'mesh_val_anom_MC_1d_awr.png')
# Only field values.
Fig, AX_dict = plot_1d.plot_con_mesh(output_1d, mesh_field={"prof": ["thetail"]}, con_field={"prof": ["w"]}, \
    YLim=[0, 1.7e3], Anomalies=False, CLim=[255, 265], cmap='plasma', format='jpg', \
    save_fig=True, fname=plot_path+'mesh_val_MC_1d_awr.jpg')
Fig, AX_dict = plot_1d.plot_con_mesh(output_1d, mesh_field={"mc": ["cpice"]}, con_field={"mc": ["cpice"]}, Suptitle=None, YLim=[0, 1.7e3], \
    Anomalies=False, CLim=[1e-15, 1e3], Levels = [x for x in np.linspace(100,1e3,4)], format='png', \
    save_fig=True, fname=plot_path+'mesh_val_MC_1d_awr2.png')
