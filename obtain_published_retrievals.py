# -*- coding: utf-8 -*-
"""
MOSiP obtain light sensor retrievals.
developped by Niels Fuchs (Universität Hamburg, 2021)
contact: niels.fuchs@uni-hamburg.de
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.tri as tri
import xarray
from matplotlib import cm
from scipy import interpolate
import xarray as xr
import os
from src import LightProfiles
from src import LightAttributes

lightharp1 = xr.load_dataset('data/processed/LM_lightharp1_200115_200718.nc')
chain_LM = xr.load_dataset('data/processed/LM_2020R11_200326_200722.nc')
chain_CO = xr.load_dataset('data/processed/CO_2020R10_200308_200323.nc')
chain_L3 = xr.load_dataset('data/processed/L3_2020R12_200424_200807.nc')
chain_Leg5 = xr.load_dataset('data/processed/Leg5_2020R21_200825_201025.nc')


# lightharp retrievals

lightharp_calibration = np.load('data/calib/calibration_coefficients_opthoharp_2022.npy') # [sensor, [VIS_in, NIR_in, VIS_out, NIR_out], [C, R,G,B]]

ds = lightharp1.copy()

for var in ds.data_vars:
    if ds[var].data.size > 1 and var != 'datenum':
        ds[var] = ds[var].astype('float32', copy=False)
        ds[var].values[ds[var].values == 65535] = np.nan


# add field with time delay between nominal date and measurement in depth y
delay_seconds = np.zeros((ds.nominal_date.shape[0], ds.depth.shape[0])).astype('float32')

for x in range(ds.nominal_date.shape[0]):
    for y in range(ds.depth.shape[0]):
        try:
            delay_seconds[x,y] = \
                ((np.array(mdates.num2date(ds.datenum.isel(nominal_date=x, depth=y))).astype('datetime64[ns]')-ds.nominal_date.isel(nominal_date=x).data)/(1e9)).astype('float32')    # calculate timedelta and convert from ns to s
        except:
            delay_seconds[x, y] = np.nan
ds['delay_from_nominal'] = (('nominal_date', 'depth'), delay_seconds.astype('float32'))
ds = ds.drop(labels='datenum')

# slightly reduce file size
for coord in ds.coords:
    if ds[coord].dtype == 'float64':
        ds[coord] = ds[coord].astype('float32',copy=False)

# prepare second dataset for layer properties

top_layer=[0,1,3,4,5,6]
bottom_layer=[1,3,4,5,6,7]

ds_layer = ds.copy()
ds_layer['sensor_id'] = xr.DataArray('lightharp1_layers', dims=(), attrs={'cf_role': 'trajectory_id'})
for var in ds_layer.data_vars:
    if ds[var].ndim > 1:
        ds_layer = ds_layer.drop(var)

ds_layer['depth_center'] = (('depth_center'), (ds.depth.data[top_layer]+ds.depth.data[bottom_layer])/2.)
ds_layer['depth_top'] = (('depth_center'), ds.depth.data[top_layer])
ds_layer['depth_bottom'] = (('depth_center'), ds.depth.data[bottom_layer])

ds_layer['ice_depth_top'] = (('depth_center', 'nominal_date'), ds.ice_depth.data[top_layer,:])
ds_layer['ice_depth_center'] = (('depth_center', 'nominal_date'),
                                (ds.ice_depth.data[top_layer,:]+ds.ice_depth.data[bottom_layer,:])/2)
ds_layer['ice_depth_bottom'] = (('depth_center', 'nominal_date'), ds.ice_depth.data[bottom_layer,:])
ds_layer = ds_layer.set_coords(('depth_top', 'depth_bottom', 'depth_center', 'ice_depth_top', 'ice_depth_bottom', 'ice_depth_center'))

ds_layer = ds_layer.drop_dims('depth')

layer_init_array = (('nominal_date', 'depth_center'),
              np.ones((ds_layer.nominal_date.shape[0], ds_layer.depth_center.shape[0]))*np.nan)

for n_spec,spec in enumerate(['VIS', 'NIR']):
    for n_b, b in enumerate(['C', 'R', 'G', 'B']):
        for n_ori, orientation in enumerate(['in', 'out']):
            ds[spec + '_' + orientation + '_' + b + '_cal'] = ds[spec + '_' + orientation + '_' + b]*lightharp_calibration[:, n_spec+n_ori*2, n_b]
        for cal in ['', '_cal']:

            # scalar irradiance
            ds[spec + '_scalar_' + b + cal] = 2. * ds[spec + '_in_' + b + cal] + \
                                        2. * ds[spec + '_out_' + b + cal]

            # slab albedo
            ds[spec + '_slab_albedo_' + b + cal] = ds[spec + '_out_' + b + cal] / ds[spec + '_in_' + b + cal]
    
            # attenuation coefficient
            ds_layer[spec + '_scalar_attenuation_' + b + cal] = layer_init_array

            ds_layer[spec + '_scalar_attenuation_' + b + cal] = (('nominal_date', 'depth_center'),-(np.log(
                ds[spec + '_scalar_' + b + cal].isel(depth=bottom_layer).data /
                ds[spec + '_scalar_' + b + cal].isel(depth=top_layer).data) /
                                                           (ds.depth.isel(depth=bottom_layer).data -
                                                            ds.depth.isel(depth=top_layer).data)))

            ds_layer[spec + '_downwelling_attenuation_' + b + cal] = layer_init_array

            ds_layer[spec + '_downwelling_attenuation_' + b + cal] = (('nominal_date', 'depth_center'), -(np.log(
                ds[spec + '_in_' + b + cal].isel(depth=bottom_layer).data /
                ds[spec + '_in_' + b + cal].isel(depth=top_layer).data) /
                                                                                                     (ds.depth.isel(
                                                                                                         depth=bottom_layer).data -
                                                                                                      ds.depth.isel(
                                                                                                          depth=top_layer).data)))

            ds_layer[spec + '_upwelling_attenuation_' + b + cal] = layer_init_array

            ds_layer[spec + '_upwelling_attenuation_' + b + cal] = (('nominal_date', 'depth_center'), -(np.log(
                ds[spec + '_out_' + b + cal].isel(depth=bottom_layer).data /
                ds[spec + '_out_' + b + cal].isel(depth=top_layer).data) /
                                                                                                     (ds.depth.isel(
                                                                                                         depth=bottom_layer).data -
                                                                                                      ds.depth.isel(
                                                                                                          depth=top_layer).data)))


            # transmissivity
    
            ds_layer[spec + '_transmissivity_' + b + cal] = layer_init_array

            ds_layer[spec + '_transmissivity_' + b + cal] = (('nominal_date', 'depth_center'),
                                                             (ds[spec + '_out_' + b + cal].isel(depth=bottom_layer).data *
                                                 ds[spec + '_out_' + b + cal].isel(depth=top_layer).data -
                                                 ds[spec + '_in_' + b + cal].isel(depth=bottom_layer).data *
                                                 ds[spec + '_in_' + b + cal].isel(depth=top_layer).data) / \
                                                                           (ds[spec + '_out_' + b + cal].isel(depth=bottom_layer).data ** 2 -
                                                                            ds[spec + '_in_' + b + cal].isel(depth=top_layer).data ** 2))
    
            # attenuation scaled
    
            #ds_layer[spec + '_attenuation_scaled_' + b + cal] = xr.full_like(ds[spec + '_in_' + b + cal], np.nan)
            #ds_layer[spec + '_attenuation_scaled_' + b + cal][:, 0:len(ds.depth) - 1] = -np.log(ds[spec + '_transmissivity_' + b + cal][:, 0:len(ds.depth) - 1])/\
            #                                                                (ds.depth.isel(depth=range(1, len(ds.depth))).data -
            #                                                ds.depth.isel(depth=range(len(ds.depth) - 1)).data)
            #
            #standard_profile = LightProfiles.IceProfile(data=ds_layer[spec + '_attenuation_scaled_' + b + cal].to_dataset(),
            #                                            id=ds.attrs['buoy_ID'],
            #                                            meta=LightAttributes.nc_meta_2_station_dict((ds.attrs).copy()),
            #                                            kind='retrieved')
            #standard_profile.write_NetCDF()
    
            # reflectivity
    
            ds_layer[spec + '_reflectivity_' + b + cal] = layer_init_array
            ds_layer[spec + '_reflectivity_' + b + cal] = (('nominal_date', 'depth_center'),
                                                                                   (ds[spec + '_out_' + b + cal].isel(depth=bottom_layer).data *
                                                 ds[spec + '_in_' + b + cal].isel(depth=bottom_layer).data -
                                                 ds[spec + '_out_' + b + cal].isel(depth=top_layer).data *
                                                 ds[spec + '_in_' + b + cal].isel(depth=top_layer).data) / \
                                                                          (ds[spec + '_out_' + b + cal].isel(depth=bottom_layer).data ** 2 -
                                                                           ds[spec + '_in_' + b + cal].isel(depth=top_layer).data ** 2))

    
            # extinction
    
            ds_layer[spec + '_absorptivity_' + b + cal] = layer_init_array
            ds_layer[spec + '_absorptivity_' + b + cal] = (('nominal_date', 'depth_center'),
                                                           1.-((ds[spec + '_out_' + b + cal].isel(depth=top_layer).data +
                                                 ds[spec + '_in_' + b + cal].isel(depth=bottom_layer).data) /
                                                 (ds[spec + '_out_' + b + cal].isel(depth=bottom_layer).data +
                                                 ds[spec + '_in_' + b + cal].isel(depth=top_layer).data)))
standard_profile = LightProfiles.IceProfile(data=ds_layer,id=ds.attrs['buoy_ID']+'_layer',meta=LightAttributes.nc_meta_2_station_dict((ds.attrs).copy()))
standard_profile.write_NetCDF()
## lightharp PAR

lin_comb2PAR = np.array([1.09894646, 0.98896817, 1.57290738])
popt_rgb_uncal_quant=np.array([0.00098748, 0.00118794, 0.00101161])
popt_rgb_cal_quant=np.array([0.00106544, 0.00123258, 0.00108111])

for cal in ['','_cal']:

    if cal:
        popt_rgb = popt_rgb_cal_quant.copy()
    else:
        popt_rgb = popt_rgb_uncal_quant.copy()

    ds['PAR_proxy_in' + cal] = xr.full_like(ds['VIS_in_C' + cal], np.nan)
    ds['PAR_proxy_out' + cal] = xr.full_like(ds['VIS_in_C' + cal], np.nan)
    ds['PAR_proxy_scalar' + cal] = xr.full_like(ds['VIS_in_C' + cal], np.nan)

    for i in range(8):
        ds['PAR_proxy_in' + cal][:,i] = np.dot(np.array([ds['VIS_in_'+band + cal].isel(depth=i).data for band in ['R', 'G', 'B']]).T*popt_rgb, lin_comb2PAR)
        ds['PAR_proxy_out' + cal][:,i] = np.dot(np.array([ds['VIS_out_'+band + cal].isel(depth=i).data for band in ['R', 'G', 'B']]).T*popt_rgb, lin_comb2PAR)
        ds['PAR_proxy_scalar' + cal][:,i] = 2*ds['PAR_proxy_in' + cal][:,i] + 2*ds['PAR_proxy_out' + cal][:,i]

standard_profile = LightProfiles.IceProfile(data=ds,id=ds.attrs['buoy_ID'],meta=LightAttributes.nc_meta_2_station_dict((ds.attrs).copy()))
standard_profile.write_NetCDF()

# light chain retrievals

for c, ds in enumerate([chain_LM.copy(), chain_CO.copy(), chain_L3.copy(), chain_Leg5.copy()]):
    for cal in ['', '_cal']:
        if ds.sensor_id == '2020R12' and cal == '_cal':
            continue
        for var in ds.data_vars:
            if ds[var].data.size > 1:
                ds[var] = ds[var].astype('float32', copy=False)
                ds[var].values[ds[var].values == 65535] = np.nan

        for b in ['C', 'R', 'G', 'B']:
            ds['VIS_scalar_' + b + cal] = 4. * ds['VIS_side_' + b + cal]

            # attenuation from level to level

            ds['VIS_scalar_attenuation_' + b + cal] = xr.full_like(ds['VIS_scalar_' + b + cal], np.nan)
            ds['VIS_scalar_attenuation_' + b + cal][:, 0:len(ds.depth)-1] = -(np.log(
                ds['VIS_scalar_' + b + cal].isel(depth=range(1, len(ds.depth))).data /
                ds['VIS_scalar_' + b + cal].isel(depth=range(len(ds.depth)-1)).data) /
                                                           (ds.depth.isel(depth=range(1, len(ds.depth))).data -
                                                            ds.depth.isel(depth=range(len(ds.depth)-1)).data))

    standard_profile = LightProfiles.IceProfile(data=ds,
                                                        id=ds.attrs['buoy_ID'],
                                                        meta=LightAttributes.nc_meta_2_station_dict((ds.attrs).copy()),
                                                        kind='retrieved')
    standard_profile.write_NetCDF()


