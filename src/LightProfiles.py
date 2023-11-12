# -*- coding: utf-8 -*-
"""
MOSiP database source code.
developped by Niels Fuchs (Universität Hamburg, 2021)
contact: niels.fuchs@uni-hamburg.de
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.tri as tri
from matplotlib import cm
from scipy import interpolate
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import xarray as xr
from src import LightAttributes

auxiliary_data_reference = {'LM': 'auxillary_LM.nc',
                            'CO': 'auxillary_CO.nc',
                            'L3': '2019T70_300234068705280_TS.csv',
                            'Leg5': '2020T81_300234068708320_TS.csv'}


def convert2xarray(data, id, **kwargs):
    """ convert pandas dataframe to xarray"""
    timeaxis = np.float64(np.unique(data['nominal_date']))
    data['nominal_date'] = np.array(mdates.num2date(data['nominal_date'])).astype('datetime64[s]')
        
    if len(np.unique(data['depth'])) > 1:
        data = data.set_index(['nominal_date', 'depth'])
    else:
        data = data.set_index(['nominal_date'])

    ds = xr.Dataset.from_dataframe(data)
    ds['sensor_id'] = xr.DataArray(id, dims=(), attrs={'cf_role': 'trajectory_id'})

    if 'lat' in list(data.columns) and 'lon' in list(data.columns):
        ds['lat'] = ('nominal_date', np.float32(data.loc[data['depth'] == data['depth'][0], 'lat']))
        ds['lon'] = ('nominal_date', np.float32(data.loc[data['depth'] == data['depth'][0], 'lon']))
    else:
        interp_lat = interpolate.interp1d(mdates.date2num(kwargs['aux']['nominal_date']),
                                          kwargs['aux']['lat'],
                                          bounds_error=False,
                                          fill_value=np.nan
                                          )
        ds['lat'] = ('nominal_date', interp_lat(timeaxis))
        interp_lon = interpolate.interp1d(mdates.date2num(kwargs['aux']['nominal_date']),
                                          kwargs['aux']['lon'],
                                          bounds_error=False,
                                          fill_value=np.nan
                                          )
        ds['lon'] = ('nominal_date', interp_lon(timeaxis))

    interp_ice = interpolate.interp1d(mdates.date2num(kwargs['aux']['nominal_date']),
                                          kwargs['aux']['ice_thickness_mean'],
                                          bounds_error=False,
                                          fill_value=np.nan
                                          )
    interp_snow = interpolate.interp1d(mdates.date2num(kwargs['aux']['nominal_date']),
                                           kwargs['aux']['snow_thickness_mean'],
                                           bounds_error=False,
                                           fill_value=np.nan
                                           )
    interp_melt = interpolate.interp1d(mdates.date2num(kwargs['aux']['nominal_date']),
                                           kwargs['aux']['surface_melt'],
                                           bounds_error=False,
                                           fill_value=np.nan
                                           )
    ds['ice_thickness'] = ('nominal_date', interp_ice(timeaxis))
    ds['snow_thickness'] = ('nominal_date', interp_snow(timeaxis))
    ds['surface_melt'] = ('nominal_date', interp_melt(timeaxis))
    ds['ice_thickness'].attrs = kwargs['aux'].attrs
    ds['snow_thickness'].attrs = kwargs['aux'].attrs
    ds['surface_melt'].attrs = kwargs['aux'].attrs

    return ds


def num2str(datenum):
    """ converts mdates datenum to datestring """
    return dt.datetime.strftime(mdates.num2date(datenum), '%Y-%m-%dT%H:%M:%S')


def dt642str(date):
    """ converts mdates datenum to datestring """
    return dt.datetime.strftime(pd.Timestamp(date).to_pydatetime(), '%Y-%m-%dT%H:%M:%S')

def dt642strfname(date):
    """ converts mdates datenum to datestring """
    return dt.datetime.strftime(pd.Timestamp(date).to_pydatetime(), '%y%m%d')


class IceProfile:
    """
    Ice Profiles are individual profile trajectories measured during the MOSAiC expedition,
    separated by variable and sensor.
    Class contains various functions for the post-processing of ice profiles including:
    - vertical axis correction
    - nominal date calculation (one date per profile measurement,
    instead of individual ones for every depth)
    - completion of missing dimensional data by sensor group
    """

    def __init__(self, **kwargs):
        """ Initialize standard profile data """

        self.meta = kwargs['meta']
        self.data = kwargs['data']
        self.meta['date_first'] = dt642str(self.data['nominal_date'].min().values)
        self.meta['date_last'] = dt642str(self.data['nominal_date'].max().values)
        #self.meta['date_first_filename'] = dt642strfname(self.data['nominal_date'].min().values)
        #self.meta['date_last_filename'] = dt642strfname(self.data['nominal_date'].max().values)
        self.id = kwargs['id']

    def correct_surface_melt(self):
        melt = SurfaceMeltRate(self.data.nominal_date.data, self.data.depth.data,
                               sensor=SurfaceMeltSupportingPoints(sensor=self.id))
        self.data.coords['ice_depth'] = (('depth', 'nominal_date'), melt.depth_array)


    def write_NetCDF(self, **kwargs):
        """ Export NetCDF file """
        # add variable attribute metadata
        self.data = LightAttributes.set_variable_attributes(self.data)
        self.data = LightAttributes.set_global_attributes(self.data, self.id, self.meta)
        self.meta['date_first_filename'] = dt642strfname(self.meta['date_first'])
        self.meta['date_last_filename'] = dt642strfname(self.meta['date_last'])
        self.data.to_netcdf('data/processed/' +
                            self.meta['site'] + '_' +
                            str(self.id) + '_' +
                            self.meta['date_first_filename'] + '_' +
                            self.meta['date_last_filename'] +
                            '.nc')



class SurfaceMeltReference:
    def __init__(self, **kwargs):
        if not kwargs:
            self.ref_time = np.array(mdates.num2date(np.array([18429.025017772074, 18430.649237947422, 18431.136504000024,
                                                      18431.753707666656, 18432.630786561345, 18433.41041224551,
                                                      18434.222522333184, 18435.002148017353, 18436.658852596207,
                                                      18438.21810396454, 18439.4200268943, 18441.174184683674,
                                                      18443.480577332666, 18445.722001174647, 18446.98889291142,
                                                      18447.703549788574, 18449.490191981455, 18452.348819490067,
                                                      18454.915087367117, 18457.156511209094, 18459.755263489653,
                                                      18461.996687331633, 18463.81581392802, 18465.147674471806,
                                                      18466.73941024365]))).astype('datetime64[ns]')
            self.ref_z = np.array([-36.58880942746049, -37.918332954035634, -50.41585410384158,
                                   -50.681758809156634, -58.92480467392227, -59.456614084552314,
                                   -56.265757620772035, -57.32937644203213, -61.849756432387494,
                                   -66.63604112805788, -69.82689759183813, -74.08137287687848,
                                   -74.6131822875085, -75.41089640345356, -80.728990509754,
                                   -86.31298932136943, -89.7697504904647, -95.08784459676511,
                                   -99.8741292924355, -104.66041398810586, -103.5967951668458,
                                   -107.85127045188611, -107.85127045188611, -108.11717515720113,
                                   -110.51031750503634]) / -100.
            self.ref_z -= self.ref_z[0]
            self.ref_buoy = '2020M29'

        else:

            self.ref_time = kwargs['ref_time']
            self.ref_z = kwargs['ref_z']
            self.ref_buoy = kwargs['ref_buoy']

        self.melt_onset = self.ref_time[0]

        self.ref_interp = self.prepare_surface_melt_solution()

    def prepare_surface_melt_solution(self):
        return interpolate.interp1d(self.ref_time.astype('float64'), self.ref_z, bounds_error=False, fill_value='extrapolate')


class SurfaceMeltSupportingPoints:
    def __init__(self,sensor=None):
        if sensor in ['lightharp1', 'saltharp1', 'saltharp2', '2020R11']:
            self.time = ['20200618120000', '20200618120000', '20200618180000',
                         '20200619120000', '20200629100000']
            self.z = (np.array([-2.0, -3.5, -6.5, -12., -23.]) / -100.) - 0.02  # top sensor exactly at ice surface
            self.min_z = 0.485  # depth of uppermost sensor that remains frozen in
            self.time = np.array([dt.datetime.strptime(b, '%Y%m%d%H%M%S')
                                  for b in self.time]).astype('datetime64[ns]')
        else:
            self.time = np.array([])




class SurfaceMeltRate:
    """
    The vertical position of frozen sensor changes, whenver ice melts and growths.
    This function corrects the vertical axis so that:
    the ice surface is at z=0,
    ice depth as positive downwards coordinate in [m],
    freeboard as additional coordinate to simply recalculate profile depth relative to freeboard.
    For every sensor group, one mass balance solution is retrieved manually as reference.
    """
    def __init__(self,
                 time_axis,
                 depth,
                 reference=SurfaceMeltReference(),
                 sensor=SurfaceMeltSupportingPoints()
                 ):

        if sensor.time.size > 0:
            # time span between onset of melting an first supporting point

            time_steps = time_axis[
                np.where(
                    np.logical_and(time_axis >= reference.melt_onset,
                                   time_axis <= sensor.time[0])
                )[0]
            ]
            time_steps_0 = time_steps
            surface_melt_0 = reference.ref_interp(time_steps.astype('float64'))

            sensor_rate = sensor.z[0]
            reference_rate = reference.ref_interp(sensor.time[0].astype('float64'))

            surface_melt_0 = surface_melt_0 * sensor_rate / reference_rate

            # during melting

            for i in range(1, len(sensor.z) - 1):
                time_steps = time_axis[
                    np.where(
                        np.logical_and(time_axis > sensor.time[i],
                                       time_axis <= sensor.time[i + 1])
                    )[0]
                ]

                time_steps_proc = time_steps
                surface_melt_proc = reference.ref_interp(time_steps.astype('float64'))

                sensor_rate = sensor.z[i + 1] - sensor.z[i]

                reference_rate = reference.ref_interp(sensor.time[i + 1].astype('float64')) \
                                 - reference.ref_interp(sensor.time[i].astype('float64'))

                surface_melt_proc = reference.ref_interp(sensor.time[i].astype('float64')) + \
                                    ((surface_melt_proc - reference.ref_interp(sensor.time[i].astype('float64'))) * \
                                     (sensor_rate / reference_rate))

                time_steps_0 = np.concatenate((time_steps_0, time_steps_proc))
                surface_melt_0 = np.concatenate((surface_melt_0, surface_melt_proc))

            # after

            time_steps = time_axis[
                np.where(time_axis > sensor.time[-1],
                         )[0]
            ]
            time_steps_proc = time_steps
            sensor_rate = sensor.z[-1] - sensor.min_z
            reference_rate = reference.ref_interp(sensor.time[-1].astype('float64')) \
                             - reference.ref_z[-1]

            surface_melt_proc = reference.ref_interp(time_steps.astype('float64'))

            surface_melt_proc = reference.ref_interp(sensor.time[-1].astype('float64')) + \
                                ((surface_melt_proc - reference.ref_interp(sensor.time[-1].astype('float64'))) * \
                                 (sensor_rate / reference_rate))

            time_steps_0 = np.concatenate((time_steps_0, time_steps_proc))
            surface_melt_0 = np.concatenate((surface_melt_0, surface_melt_proc))

        else:
            surface_melt_0 = reference.ref_interp((time_axis[time_axis >= reference.melt_onset]).astype('float64'))

        self.depth_array = np.ones((len(depth), len(time_axis))) * depth[:, np.newaxis]
        self.depth_array[:, time_axis >= reference.melt_onset] = self.depth_array[:,
                                                                 time_axis >= reference.melt_onset] - surface_melt_0[:]
