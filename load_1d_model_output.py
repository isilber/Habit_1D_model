import numpy as np
import xarray as xr
import pandas as pd
import glob, os

"""
-----------------------------------------------------------------------------------
                Israel Silber
            Last update: 7/16/2020
----------------------------------------------------------------------------------
Current methods:
    load_1d_model_output - load multiple fields from multiple output types onto an
    Xarray dataset object.
    init_load_1d_model_output - initialize model output metadata (should match the
    1D model code - updated as of 7/16/2020.
"""

#---------------------------------------------------------------------------------------------------------------

def load_1d_model_output(run_path, time_range=[np.nan, np.nan], 
           types_to_load=["prof", "flx", "tke", "mc", "sfc"], fields_to_load={}, **kwargs):
    """
    ----------------------
    Israel Silber
    Last update: 7/16/2020
    ----------------------
    This method loads (Golaz) 1D model output. The method has default output lists and
    descriptions, but can accept different output parameter sorting or lists (shoudl
    match the model output subroutines, all located in /model/initialize.f

    Parameters
    ----------
    run_path:
        path where output files are located.
    time_range:
        2 element list providing the time range in fractional hours of simulation time
        (set both to np.nan to load everything).
    type_to_load (load when true):
        list containing output types to load (currently working with prof, flx, tke,
        mc, and sfc.
    fields_to_load:
        a dictionary containing field names to load for every requested output type
        (load all fields if empty, if output type not in keys, or if output type values
        are empty).

    Returns
    -------
    Output_1D:
        xarray dataset gathering all model output types (if requested)
    """

    """
    os.chdir("/home/meteo/ixs34/Python/1d_model")
    run_path = '/home/meteo/ixs34/golaz_1d/runs/jyhRuns/results/awr_drz/'
    """
    if not kwargs:
        out_info = init_load_1d_model_output()
    else:
        out_info = init_load_1d_model_output(kwargs)

    # -------------------------------------load data------------------------------------
    output_1d = {}
    for out_type in types_to_load:
        for filename in sorted(glob.glob(run_path + out_type + "*.dat")):
            if (float(filename[-9:-6]) + float(filename[-6:-4])/60. >= time_range[0] and \
                  float(filename[-9:-6]) + float(filename[-6:-4])/60. < time_range[1]) or \
                  np.isnan(time_range[0]):
                output_tmp = pd.read_csv(filename, delim_whitespace=True, header=None,
                        names=out_info[out_type]["names"])
                # grab the model domain height array
                if "zt" in output_tmp.keys():
                    hgt = output_tmp["zt"].values.tolist()
                    output_tmp = output_tmp.drop(columns=['zt'])
                if "zm" in output_tmp.keys():
                    hgt = output_tmp["zm"].values.tolist()
                    output_tmp = output_tmp.drop(columns=['zm'])
                if len(fields_to_load) > 0:
                    if out_type in fields_to_load.keys():
                        if len(fields_to_load[out_type]) > 0:
                            fields_to_filter = []
                            for ff in fields_to_load[out_type]:
                                if ff in output_tmp.keys() and ff is not "zt" and ff is not "zm":
                                    fields_to_filter.append(ff) 
                                else:
                                    print(ff + " not in " + out_type + " output type")
                            if fields_to_filter: # filter empty lists
                                output_tmp = output_tmp.filter(items=fields_to_filter)

                # convert from pandas to xarray add append to time dim
                output_tmp = output_tmp.to_xarray()
                output_tmp = output_tmp.rename({'index': 'height'})
                output_tmp = output_tmp.expand_dims({"time": \
                        [float(filename[-9:-6]) + float(filename[-6:-4])/60.]}, axis=1)
                output_tmp = output_tmp.assign_coords({"height": hgt})

                # add attributes.
                for nn in out_info[out_type].keys():
                    if nn is not "names":
                        for ff in out_info[out_type][nn].keys():
                            if ff in output_tmp.keys():
                                output_tmp[ff] = \
                                      output_tmp[ff].assign_attrs({nn: out_info[out_type][nn][ff]})
               
                # concatenate and arrange
                if out_type not in output_1d.keys():
                    output_1d[out_type] = output_tmp
                else:
                    output_1d[out_type] = xr.concat([output_1d[out_type], output_tmp], "time")
                print("done loading " + filename)
                del output_tmp

            else:
                print(filename + " outside the requested time range (%.2f-%.2f h)" \
                      % (tuple(time_range)))

    return output_1d

#---------------------------------------------------------------------------------------------------------------

def init_load_1d_model_output(**kwargs):
    """
    This method checks whether output field["names"] other then the default were requested
    and sets them accordingly (note that if names are different, "long_name" and "units" 
    should be different as well).
    """
    out_info = {}
    # -------------------------------------init prof------------------------------------
    out_info["prof"] = {}
    if 'out_prof_names' in kwargs.keys():
        out_info["prof"]["names"] = kwargs['out_prof_names']
    else:
        out_info["prof"]["names"] = ["zm", "thetail", "u", "v", "Km", "Kh", "thv_flx", "TKE",
                            "Khh", "lx", "rt", "rl", "ri", "CF", "T", "p", "rho",
                            "thv", "htrt", "htrt_lw", "htrt_sw", "w", "hiv", "h"]
    if 'out_prof_units' in kwargs.keys():
        out_info["prof"]["units"] = kwargs['out_prof_units']
    else:
        out_info["prof"]["units"] = {"zm": "m", "thetail": "K", "u": "m/s", "v": "m/s", "Km": "", 
                            "Kh": "", "thv_flx": "J m^{-3}", "TKE": "m^2 s^{-2}", "Khh": "",
                            "lx": "m", "rt": "kg/kg", "rl": "kg/kg", "ri": "kg/kg", "CF": "", 
                            "T": "K", "p": "Pa", "rho": "kg m^{-3}",
                            "thv": "K", "htrt": "K/h", "htrt_lw": "K/h", 
                            "htrt_sw": "K/h", "w": "m/s", "hiv": "J", "h": "J"}
    if 'out_prof_long_name' in kwargs.keys():
        out_info["prof"]["long_name"] = kwargs['out_prof_long_name']
    else:
        out_info["prof"]["long_name"] = {"zm": "Mean grid height", 
                                "thetail": "Mean ice liquid potential temperature",
                                "u": "u wind", "v": "v wind", "Km": "Momentum eddy diffusivity",
                                "Kh": "Heat eddy diffusivity", "thv_flx": "Cp*rho*thvw",
                                "TKE": "Turbulent kinetik energy", 
                                "Khh": "Coefficient for second order terms",
                                "lx": "Master length scale variable for Galperin's MY2.5 scheme",
                                "rt": "Total mixing ratio", "rl": "Liquid mixing ratio (cld+rain)",
                                "ri": "Ice mixing raito", "CF": "Fractional cloudiness", 
                                "T": "Temperature", "p": "Pressure", "rho": "Density", 
                                "thv": "Virtual potential temperature",
                                "htrt": "Radiative heating rate", 
                                "htrt_lw": "LW radiative heating rate",
                                "htrt_sw": "SW radiative heating rate", "w": "vertical velocity",
                                "hiv": "ice vapor static energy", "h": "moist static energy"}
    if 'out_prof_scale' in kwargs.keys():
        out_info["prof"]["scale"] = kwargs['out_prof_scale']
    else:
        out_info["prof"]["scale"] = {"zm": "linear", "thetail": "linear", "u": "linear", 
                            "v": "linear", "Km": "linear",
                            "Kh": "linear", "thv_flx": "linear", "TKE": "linear", 
                            "Khh": "linear", "lx": "linear",
                            "rt": "log", "rl": "log", "ri": "log", "CF": "linear",
                            "T": "linear", "p": "log", "rho": "linear",
                            "thv": "linear", "htrt": "linear", "htrt_lw": "linear",
                            "htrt_sw": "linear", "w": "linear", "hiv": "linear", 
                            "h": "linear"}
    # -------------------------------------init flux------------------------------------
    out_info["flx"] = {}
    if 'out_flx_names' in kwargs.keys():
        out_info["flx"]["names"] = kwargs['out_flx_names']
    else:
        out_info["flx"]["names"] = ["zt", "Km", "Kh", "Khh", "u_turb_flx", "v_turb_flx", "thil_heat_flx",
                            "rt_heat_flx", "thv_flx", "swfu", "swfd", "lwfu", "lwfd", "SST"]
    if 'out_flx_units' in kwargs.keys():
        out_info["flx"]["units"] = kwargs['out_flx_units']
    else:
        out_info["flx"]["units"] = {"zt": "m", "Km": "", "Kh": "", "Khh": "", "u_turb_flx": "s^{-1}", 
                                    "v_turb_flx": "s^{-1}", "thil_heat_flx": "J m^{-4}",
                                    "rt_heat_flx": "J K^{-1} m^{-4}", "thv_flx": "J m^{-3}",
                                    "swfu": "W m^{-2}", "swfd": "W m^{-2}", "lwfu": "W m^{-2}",
                                    "lwfd": "W m^{-2}", "SST": "K"}
    if 'out_flx_long_name' in kwargs.keys():
        out_info["flx"]["long_name"] = kwargs['out_flx_long_name']
    else:
        out_info["flx"]["long_name"] = {"zt": "Turbulent grid heights", "Km": "Momentum eddy diffusivity",
                                "Kh": "Heat eddy diffusivity", 
                                "Khh": "Coefficient for second order terms", 
                                "u_turb_flx": "u momentum turbulent flux", 
                                "v_turb_flx": "v momentum turbulent flux",
                                "thil_heat_flx": "thetail heat flux",
                                "rt_heat_flx": "rt heat flux", 
                                "thv_flx": "Cp*rho*thvw", "swfu": "Upwelling SW flux",
                                "swfd": "Downwelling SW flux",
                                "lwfu": "Upwelling LW flux", "lwfd": "Downwelling LW flux",
                                "SST": "Surface temperature"}
    if 'out_flx_scale' in kwargs.keys():
        out_info["flx"]["scale"] = kwargs['out_flx_scale']
    else:
        out_info["flx"]["scale"] = {"zt": "linear", "Km": "linear", "Kh": "linear", 
                                    "Khh": "linear", "u_turb_flx": "linear",
                                    "v_turb_flx": "linear", "thil_heat_flx": "linear",
                                    "rt_heat_flx": "symlog", "thv_flx": "linear",
                                    "swfu": "linear", "swfd": "linear", "lwfu": "linear",
                                    "lwfd": "linear", "SST": "linear"}
# -------------------------------------init TKE------------------------------------
    out_info["tke"] = {}
    if 'out_tke_names' in kwargs.keys():
        out_info["tke"]["names"] = kwargs['out_tke_names']
    else:
        out_info["tke"]["names"] = ["zt", "TKE", "tke_shear", "tke_buoy", "tke_ttrspt", "tke_vtrspt",
                            "tke_dissip"]
    if 'out_tke_units' in kwargs.keys():
        out_info["tke"]["units"] = kwargs['out_tke_units']
    else:
        out_info["tke"]["units"] = {"zt": "m", "TKE": "m^2 s^{-2}", "tke_shear": "m^2 s^{-2}",
                        "tke_buoy": "m^2 s^{-2}",
                        "tke_ttrspt": "m^2 s^{-2}", "tke_vtrspt": "m^2 s^{-2}", "tke_dissip": "m^2 s^{-2}"}
    if 'out_tke_long_name' in kwargs.keys():
        out_info["tke"]["long_name"] = kwargs['out_tke_long_name']
    else:
        out_info["tke"]["long_name"] = {"zt": "Turbulent grid heights", "TKE": "Turbulent kinetik energy",
                                "tke_shear": "TKE shear production",
                                "tke_buoy": "TKE buoyancy production",
                                "tke_ttrspt": "TKE turbulent transport",
                                "tke_vtrspt": "TKE vertical transport",
                                "tke_dissip": "TKE dissipation"}
    if 'out_tke_scale' in kwargs.keys():
        out_info["tke"]["scale"] = kwargs['out_tke_scale']
    else:
        out_info["tke"]["scale"] = {"zt": "linear", "TKE": "linear", "tke_shear": "linear",
                                    "tke_buoy": "linear",
                                    "tke_ttrspt": "linear", "tke_vtrspt": "linear",
                                    "tke_dissip": "linear"}
    # -------------------------------------init microphysics---------------------------------
    out_info["mc"] = {}
    if 'out_mc_names' in kwargs.keys():
        out_info["mc"]["names"] = kwargs['out_mc_names']
    else:
        out_info["mc"]["names"] = ["zm", "rt", "rvap", "rcld", "rrain", "rpice", "rsnow", "raggr",
                           "rgrau", "rhail", "cpice", "amean", "cmean", "rhoiavg", "aspect",
                           "vthabn", "vthabm", "dmtvap" ,"ssi", "drpp_dep", "drpp_sedim",
                           "drrp_sedim", "drrp_collect", "drt_horizAdv", "drc_ncphys",
                           "drv_mcphys", "drc_turb", "drv_turb", "dthil_subs", "drt_subs"]
    if 'out_mc_units' in kwargs.keys():
        out_info["mc"]["units"] = kwargs['out_mc_units']
    else:
        out_info["mc"]["units"] = {"zm": "m", "rt": "kg/kg", "rvap": "kg/kg", "rcld": "kg/kg",
                           "rrain": "kg/kg", "rpice": "kg/kg", "rsnow": "kg/kg", "raggr": "kg/kg",
                           "rgrau": "kg/kg", "rhail": "kg/kg", "cpice": "m^{-3}", "amean": "\mu m",
                           "cmean": "\mu m", "rhoiavg": "kg m^{-3}", "aspect": "", "vthabn": "m/s",
                           "vthabm": "m/s", "dmtvap": "-" ,"ssi": "-",
                           "drpp_dep": "kg kg^{-1} s^{-1}", "drpp_sedim": "kg kg^{-1} s^{-1}",
                           "drrp_sedim": "kg kg^{-1} s^{-1}", "drrp_collect": "kg kg^{-1} s^{-1}",
                           "drt_horizAdv": "kg kg^{-1} s^{-1}", "drc_ncphys": "kg kg^{-1} s^{-1}",
                           "drv_mcphys": "kg kg^{-1} s^{-1}", "drc_turb": "kg kg^{-1} s^{-1}",
                           "drv_turb": "kg kg^{-1} s^{-1}", "dthil_subs": "K s^{-1}",
                           "drt_subs": "kg kg^{-1} s^{-1}"}

    if 'out_mc_long_name' in kwargs.keys():
        out_info["mc"]["long_name"] = kwargs['out_mc_long_name']
    else:
        out_info["mc"]["long_name"] = {"zm": "Mean grid height", "rt": "Total mixing ratio",
                               "rvap": "Vapor mixing ratio", "rcld": "Cloud water mixing ratio",
                               "rrain": "Rain water mixing ratio",
                               "rpice": "Pristine ice mixing ratio", "rsnow": "Snow mixing ratio",
                               "raggr": "Aggregate mixing raito",
                               "rgrau": "graupel mixing ratio", "rhail": "hail mixing ratio",
                               "cpice": "Ice number concentration",
                               "amean": "mean a-axis length", "cmean": "mean a-axis length",
                               "rhoiavg": "Mean ice density", "aspect": "Aspect ratio",
                               "vthabn": "Number weighted terminal fall speed",
                               "vthabm": "Mass weighted terminal velocity",
                               "dmtvap": "-", "ssi": "Ice supersaturation",
                               "drpp_dep": "Tendencies for depositional growth of pristine ice",
                               "drpp_sedim": "Tendencies for sedimentation of pristine ice",
                               "drrp_sedim": "Tendencies for sedimentation of rain",
                               "drrp_collect": "Tendencies for collection of cloud into rain",
                               "drt_horizAdv": "Tendencies in rt due to horizontal advection",
                               "drc_ncphys": "Tendencies in rc due to microphysics",
                               "drv_mcphys": "Tendencies in rv due to microphysics",
                               "drc_turb": "Tendencies in rc due to turbulence",
                               "drv_turb": "Tendencies in rv due to turbulence",
                               "dthil_subs": "Tendencies in thetail due to subsidence",
                               "drt_subs": "Tendencies in rt due to subsidence"}

    if 'out_mc_scale' in kwargs.keys():
        out_info["mc"]["scale"] = kwargs['out_mc_scale']
    else:
        out_info["mc"]["scale"] = {"zm": "linear", "rt": "linear", "rvap": "linear",
                           "rcld": "log",
                           "rrain": "log", "rpice": "log", "rsnow": "log", "raggr": "log",
                           "rgrau": "log", "rhail": "log", "cpice": "log", "amean": "log",
                           "cmean": "log", "rhoiavg": "linear", "aspect": "linear",
                           "vthabn": "linear",
                           "vthabm": "linear", "dmtvap": "linear" ,"ssi": "linear",
                           "drpp_dep": "symlog", "drpp_sedim": "log",
                           "drrp_sedim": "log", "drrp_collect": "log",
                           "drt_horizAdv": "symlog", "drc_ncphys": "symlog",
                           "drv_mcphys": "symlog", "drc_turb": "symlog",
                           "drv_turb": "symlog", "dthil_subs": "linear",
                           "drt_subs": "symlog"}
    # -------------------------------------init sfc------------------------------------
    out_info["sfc"] = {}
    if 'out_sfc_names' in kwargs.keys():
        out_info["sfc"]["names"] = kwargs['out_sfc_names']
    else:
        out_info["sfc"]["names"] = ["time", "SHF", "LHF", "swd", "swu", "lwd", "lwu", "Qi", "SST"]
    if 'out_sfc_units' in kwargs.keys():
        out_info["sfc"]["units"] = kwargs['out_sfc_units']
    else:
        out_info["sfc"]["units"] = {"time": "s", "SHF": "W m^{-2}", "LHF": "W m^{-2}", "swd": "W m^{-2}",
        "swu": "W m^{-2}", "lwd": "W m^{-2}", "lwu": "W m^{-2}", "Qi": "W m^{-2}", "SST": "K"}
    if 'out_sfc_long_name' in kwargs.keys():
        out_info["sfc"]["long_name"] = kwargs['out_sfc_long_name']
    else:
        out_info["sfc"]["long_name"] = {"time": "time", "SHF": "Sensible heat flux",
        "LHF": "Latent heat flux", "swd": "Downwelling SW flux", "swu": "Upwelling SW flux",
        "lwd": "Downwelling LW flux", "lwu": "Upwelling LW flux",
        "Qi": "Heat flux from ocean into ice", "SST": "Surface temperature"}
    if 'out_sfc_scale' in kwargs.keys():
        out_info["sfc"]["scale"] = kwargs['out_sfc_scale']
    else:
        out_info["sfc"]["scale"] = {"time": "linear", "SHF": "linear", "LHF": "linear",
                                    "swd": "linear",
                                    "swu": "linear", "lwd": "linear", "lwu": "linear",
                                    "Qi": "linear", "SST": "linear"}

    return out_info
